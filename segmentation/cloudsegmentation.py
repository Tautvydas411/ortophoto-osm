#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 31 14:59:26 2020

@author: as
"""

import sys, os, time;


os.environ["HDF5_DISABLE_VERSION_CHECK"] = "1"

import hashlib, argparse;
from mpi4py import MPI
import numpy as np, math;
import shapely.wkt, shapely.geometry, shapely.ops;
import logging;
import mxnet as mx;
from mxnet import gluon;
import threading
from mxnet.gluon.data.vision import transforms
from gluoncv.utils.parallel import DataParallelModel;
from gluoncv.model_zoo.segbase import *
from multiprocessing import Process as MProcess, Queue as MQueue, Event as MEvent;
import queue;
import boto3;
from urllib.parse import urlparse
import pickle
import io, PIL.Image;
import tqdm;

sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+"/../")
from pictureslib import  orotogis, ortophoto, CluodStorage;


import gluoncv;


gluoncv.data.datasets['osm-vdd'] = CluodStorage.CloudSegmentationDataSetV2

periods = {
    'I' : '/mnt/mosaic/period-1b.vrt',
    'II' : '/mnt/mosaic/period-2b.vrt',
    'III' : '/mnt/mosaic/period-3b.vrt',
}


UPDATE_TAG = 42


class CloudMosaicAccess(mx.gluon.data.Dataset):
    
    
    __slots__ = ["_tiledir","_onetile","_scale","vkt_url","grid_list","period_vrt","initalized","geotransform"];
     

    def __getstate__(self):
        state = self.__dict__.copy();
        if '_onetile' in state:    del state['_onetile'];
        if '_tiledir' in state:    del state['_tiledir'];
        if '_scale'   in state:    del state['_scale'];
        return state;

    def __setstate__(self, state):
        self.__dict__.update(state)
        #self._opengeotiff();

        
    def _opengeotiff(self):
        self._tiledir = ortophoto.TileDirectory();
        self._onetile = self._tiledir.open(self.period_vrt);
        
        t = self._onetile.GetGeoTransform();
        
        self.geotransform = t;

        #warning: by default scale factor is assumed 0.5
        if (t[1] == 0.5):
            self._scale = 1;
        else:
            self._scale = (t[1] / 0.5 )
        self.initalized = True;
        
    def __init__(self, grid_list, period_vrt : str = periods['I'] ):
        super(CloudMosaicAccess,self).__init__();
        
        self.grid_list = grid_list;
        self.period_vrt = period_vrt;
        self.initalized = False;
        self.input_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
        ])

        
      
    def GetGeoTransform(self):
        if (self.initalized == False):
            self._opengeotiff();
        return self.geotransform;

    def getScale(self):
        if (self.initalized == False):
           self._opengeotiff();
        return self._scale;

    def __len__(self):
        return len(self.grid_list);

    def __getitem__(self, idx):
        """
        Get dataset item
        (np.array(picture),pixelcoordinates_tuple(xoff,yoff,w,h))
        """
        if (self.initalized == False):
            self._opengeotiff();

            import ctypes, gdal;
            from ctypes import create_string_buffer
            libc = ctypes.cdll.LoadLibrary('libc.so.6')
            title = "s3worker-%d" % os.getpid() ;
            libc.prctl(15,create_string_buffer(title.encode()),0,0,0);

            tmax=min(2,len(os.sched_getaffinity(0))); # OpenCV bug?, limiting to 8 threads

            cproc = ctypes.cdll.LoadLibrary("libgomp.so.1");
            cproc.omp_set_num_threads(ctypes.c_int(1))
            gdal.SetCacheMax(6*1024*1024*1024);
            gdal.SetConfigOption('CPL_VSIL_CURL_CHUNK_SIZE', '33554432')
            gdal.SetConfigOption('GDAL_INGESTED_BYTES_AT_OPEN','262144');


        #print("%r" % (os.getpid(),self._data[idx]))
        
        box = self.grid_list[idx];
        
        if (box == None): # the last batch
            z = np.zeros((1024,1024,3),dtype='float32');
            return (self.input_transform(mx.nd.array(z)),(mx.nd.array(z,dtype='float32'), mx.nd.array((-1,-1,-1,-1),dtype='int64')))
        
        p1 = self._onetile.Coord3346ToPixel(box.bounds[0], box.bounds[1])
        p2 = self._onetile.Coord3346ToPixel(box.bounds[2],box.bounds[3]);
        
        x = int(p1[0]);
        y = int(p2[1]);
        
        w = int(p2[0] - p1[0]);
        h = int(p1[1] - p2[1]); # the y axis is negative

        #print("Getting %d %d %d %d" % (x,y,w,h));

        rgb = self._onetile.GetRGB(x,y,w,h);
        if (self._scale != 1): # TODO: watch this...
           import cv2;
           rgb = cv2.resize(rgb,dsize=(1024,1024), interpolation=cv2.INTER_CUBIC);

        
        a = rgb.astype('float32');

        #print("shapas %r %s %r -- %r %r .. %r %r" % (a.shape,a.dtype,self._scale,np.isnan(a).any(),np.isinf(a).any(),a.max(),a.min()));
            
        p1 = np.percentile(a,2)
        p2 = np.percentile(a,98);
            
        a[ a < p1] = p1;
        a[ a > p2] = p2;
            
        a = a - p1;
        if (p2 -p1 ) == 0:
            a = a * 0;
        else:
            a = a / (p2 -p1);

        return (self.input_transform(mx.nd.array(a)),(mx.nd.array(rgb,dtype='float32'), mx.nd.array((x,y,w,h),dtype='int64')))
        


def UploaderFunction(args : dict, worker_num : int):
        log = logging.Logger("Uploader messages");
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s');
        log.addHandler(logging.StreamHandler())
        logging.StreamHandler().setFormatter(formatter)

        log.info("starting %d s3 uploader " % worker_num );
        
        if (sys.platform == 'linux'):
            import ctypes, gdal;
            from ctypes import create_string_buffer
            libc = ctypes.cdll.LoadLibrary('libc.so.6')
            title = "s3uploader-%02d" % worker_num ;
            libc.prctl(15,create_string_buffer(title.encode()),0,0,0);

            tmax=min(2,len(os.sched_getaffinity(0))); # OpenCV bug, limiting to 8 threads?

            cproc = ctypes.cdll.LoadLibrary("libgomp.so.1");
            cproc.omp_set_num_threads(ctypes.c_int(1))
            gdal.SetCacheMax(6*1024*1024*1024)
            gdal.SetConfigOption('CPL_VSIL_CURL_CHUNK_SIZE', '33554432')
            gdal.SetConfigOption('GDAL_INGESTED_BYTES_AT_OPEN','262144');
         
        session = boto3.session.Session()
        s3_client = session.client(
                service_name='s3',
                aws_access_key_id='xxxx',
                aws_secret_access_key='xxxx',

                endpoint_url='https://s3.litnet.lt',
          )

        q = args['mp_q'];
        evt = args['mp_e'];

        gt = args['geotransform'];

        s3_bucket = args['s3_bucket'];
        s3_bucket_prefix = args['s3_bucket_prefix'];

   
        while not mp_e.is_set():
            try:
              item = q.get(block=True,timeout=1);

              if (item['coords'][0] == -1):
                log.warning("empty image received, ignoring");
                continue;
              area_prefix = item['area'];

              coords_str = "%08d-%08d-%06d-%06d" % tuple(item['coords']);


              filename= "%s/%s/%s.png" % (s3_bucket_prefix,area_prefix,coords_str);
              filename_wld = "%s/%s/%s.wld" % (s3_bucket_prefix,area_prefix,coords_str)

              pictureio    = io.BytesIO();
              picturewldio = io.StringIO();


              # create picture

              results_img = PIL.Image.fromarray(item['result'],mode='P');
              results_img.putpalette(orotogis.prefered_color_u8_flat_short);
              results_img.save(pictureio,format="PNG")

              # write world file

              xoff, a, b, yoff, d, e = gt;

              xp = a * item['coords'][0] + b * item['coords'][1] + a * 0.5 + b * 0.5 + xoff
              yp = d * item['coords'][0] + e * item['coords'][1] + d * 0.5 + e * 0.5 + yoff


              picturewldio.write("%f\n" % 0.5); # pixel x size
              picturewldio.write("%f\n" % 0.0);
              picturewldio.write("%f\n" % 0.0);
              picturewldio.write("%f\n" % -0.5); # pixel y size
              picturewldio.write("%f\n" % xp); # x coordinate
              picturewldio.write("%f\n" % yp); # y coordinate


              #print("Writing %s and %s" % (filename,filename_wld))

              picturewldio.seek(0);
              pictureio.seek(0);
              bpicturewldio = io.BytesIO(picturewldio.read().encode('utf8'))


              s3_client.upload_fileobj(bpicturewldio,s3_bucket,filename_wld);
              s3_client.upload_fileobj(pictureio,s3_bucket,filename);

            except queue.Empty:
              continue;


if __name__ == '__main__':
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()

    print("Starting process %d/%d\n" % (rank,nprocs));
    parser = argparse.ArgumentParser(description='The segmentation for vdd project',add_help=False);
     
    parser.add_argument('--period',choices=['I','II','III'],nargs='+',required=True,
                        help="Date period I - 2009-2010 II - 2012-2013 III- 2015-2016-2017");
     
    parser.add_argument('--params_file',type=str,required=True,
                        help="network parameters file");
     
     
     
    parser.add_argument('--s3-base-url',type=str,required=True,help='base url to s3 storage bucket and prefix  s3://test/a1')
    
    parser.add_argument('--test-batch-size', type=int, default=mx.context.num_gpus(),
                        metavar='N', help="input batch size for \
                        testing (default: %d)" % mx.context.num_gpus())

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')



    parser.add_argument('--ngpus', type=int,
                        default=mx.context.num_gpus(),
                        help="number of GPUs (default: %d)" % mx.context.num_gpus())


    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')

    parser.add_argument('--model', type=str, default='deeplab',
                        help='model name (default: deeplab)')

    parser.add_argument('--backbone', type=str, default='resnet50',
                        help='backbone name (default: resnet50)')


    parser.add_argument('--syncbn', action='store_true', default=False,
                        help='using Synchronized Cross-GPU BatchNorm')

    parser.add_argument('--dataset', type=str, default='osm-vdd',
                        help='dataset name (default: osm-vdd)')
    parser.add_argument('--aux', action='store_true', default=False,
                        help='Auxiliary loss')
    parser.add_argument('--aux-weight', type=float, default=0.5,
                        help='auxiliary loss weight')

    parser.add_argument('--base-size', type=int, default=1024,
                        help='base image size -- model param')
    parser.add_argument('--crop-size', type=int, default=1024,
                        help='crop image size -- model param')

    parser.add_argument('--strip-corner-pixels',type=int,default=0,
                        help='strip pixels arounds image for ignoring misdetection around corners')

    parser.add_argument('--dtype', type=str, default='float32',
                        help='data type for training. default is float32')



    group_wkt_parser = argparse.ArgumentParser(add_help=False);
    group_file_wkt = group_wkt_parser.add_argument_group('WKT area')

    group_file_wkt.add_argument('--wkt-areas-file', action="store", type=str, required = True,help='CSV file containing area polygon or use \'SQL\' for pgsql query')
    group_file_wkt.add_argument('--wkt-area-id', action="store", type=int, required = True,help='ID of area polygon in SQL case query:\nselect   m2.id,m2.title, ST_AsEWKB(ST_Transform(way,3346)) from planet_osm_polygon pop, municipalities m2  where pop.osm_id  = m2.osm_id  and m2.id = ?;\n ')



    parser.add_argument_group()



    #group1 = parser.add_mutually_exclusive_group(required=True)
    
    parserc = argparse.ArgumentParser(parents=[parser, group_wkt_parser])
    args = parserc.parse_args();

    # Synchronized BatchNorm
    args.norm_layer = mx.gluon.contrib.nn.SyncBatchNorm if args.syncbn \
        else mx.gluon.nn.BatchNorm
    args.norm_kwargs = {'num_devices': args.ngpus} if args.syncbn else {}


    o = urlparse(args.s3_base_url, allow_fragments=False)

    args.s3_bucket = o.netloc;
    args.s3_bucket_prefix = o.path.strip("/");

    # handle contexts
    if args.no_cuda:
        print('Using CPU')
        args.kvstore = 'local'
        args.ctx = [mx.cpu(0)]
    else:
        print('Number of GPUs:', args.ngpus)
        args.ctx = [mx.gpu(i) for i in range(args.ngpus)]


    if rank == 0:
        
        log = logging.Logger("General messages");
        formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s');

        log.addHandler(logging.StreamHandler())
        logging.StreamHandler().setFormatter(formatter)
        if (args.wkt_areas_file.upper() == 'SQL'):
          hp = orotogis.HomeProcess();

          c = hp.conn.cursor();
          s = c.execute("select m2.id as id ,m2.title as title , ST_AsEWKB(ST_Transform(way,3346)) as polygon from planet_osm_polygon pop, municipalities m2  where pop.osm_id  = m2.osm_id  and m2.id = %s;",(args.wkt_area_id,));
          f = c.fetchall();

          areas_wkt = [];
          for i in f:
            areas_wkt.append(i);


        else:
          areas_wkt = np.genfromtxt(args.wkt_areas_file,dtype=None,delimiter='\t',names=True,encoding='utf8');

        areas_wkt_dict = {};
        
        for l in range(len(areas_wkt)):
            areas_wkt_dict[ areas_wkt[l][0]] = areas_wkt[l];
            
        area_csv = areas_wkt_dict[args.wkt_area_id]
        
        log.info("Parsing '%s' area information" % area_csv[1]); 
        try:
          area_shape = shapely.wkt.loads(area_csv[2]); # try as string
        except:
          area_shape = shapely.wkb.loads(bytes(area_csv[2])); # if failure it is binary data
        
        minx = math.floor(area_shape.bounds[0]) - (args.strip_corner_pixels * 0.5)
        miny = math.floor(area_shape.bounds[1]) - (args.strip_corner_pixels * 0.5)
        
        maxx = math.floor(area_shape.bounds[2])  + (args.strip_corner_pixels * 0.5)
        maxy = math.floor(area_shape.bounds[3])  + (args.strip_corner_pixels * 0.5)

        xcursor = minx;
        ycursor = miny;

        xstep = 1024 * 0.5  - ( 2 * (args.strip_corner_pixels * 0.5))
        ystep = 1024 * 0.5  - ( 2 * (args.strip_corner_pixels * 0.5));
        
        xwidth = 1024 * 0.5;
        yheight = 1024 * 0.5;

        data_collection = [];
        while (ycursor < maxy):
            xcursor = minx;
            while (xcursor < maxx):

                g1 = shapely.geometry.box(xcursor, ycursor, xcursor + xwidth, ycursor + yheight);
                if (g1.intersects(area_shape)):
                    data_collection.append(g1);
                xcursor += xstep;
            ycursor += ystep;
        log.info("Done generation of processing rectangles - %d areas generated" % len(data_collection)); 


        data_collection = data_collection;
        ave, res = divmod(len(data_collection), nprocs)
        mpicount = [ave + 1 if p < res else ave for p in range(nprocs)]
        mpicount = np.array(mpicount,dtype=np.int)
        # displacement: the starting index of each sub-task
        mpidispl = [sum(mpicount[:p]) for p in range(nprocs)]
        mpidispl = np.array(mpidispl)
        
        data_collection_array = [];
        for mpii in range(len(mpicount)):
            data_collection_array.append(data_collection[ mpidispl[mpii]:mpidispl[mpii]+mpicount[mpii]]);
    else:
            # initialize count on worker processes
            data_collection_array = None;
            mpicount = np.zeros(nprocs, dtype=np.int)
            mpidispl = None


    comm.Barrier()

    # broadcast count
    comm.Bcast(mpicount, root=0)

    recvbuf = np.zeros(mpicount[rank],dtype=object)

    # interleave array for more optimal access 
    data_collection_array = np.array(comm.scatter(data_collection_array, root=0),dtype=object);
    if (len(data_collection_array) % args.workers != 0):
        data_collection_array = np.hstack((data_collection_array, [None] * ( args.workers - (len(data_collection_array) % args.workers)) ))
    
    data_collection_array = data_collection_array.reshape(args.workers,-1).reshape((-1,),order='F')

    # fillup for divisible batch
    
    if (len(data_collection_array) % args.test_batch_size != 0):
        data_collection_array = np.hstack((data_collection_array, [None] * ( args.test_batch_size - (len(data_collection_array) % args.test_batch_size)) ))


    assert len(args.period) == 1

    valset = CloudMosaicAccess(data_collection_array, period_vrt = periods[args.period[0]])

    eval_data = gluon.data.DataLoader(valset,
                            args.test_batch_size,
                            last_batch='keep',
                            num_workers=args.workers,
                            prefetch = args.workers * 8,
                            thread_pool = False,pin_memory = True, timeout = 600)

    if (rank == 0):
        print(args);
        sys.stdout.flush();
        sys.stderr.flush();

        tbar = tqdm.tqdm(total=np.sum(mpicount) // args.test_batch_size,unit='batch')

    else:
        tbar = None; # all other ranks



    pixel_scale = valset.getScale();
    ### Setting multiprocess 
    mp_q = MQueue(256);
    mp_e =  MEvent();
    
    
    params_dict  = {'mp_e' : mp_e, 
                    'mp_q' : mp_q,
                    'geotransform' : valset.GetGeoTransform(),
                    's3_bucket' : args.s3_bucket,
                    's3_bucket_prefix' : args.s3_bucket_prefix,
                   }
    
    mp_pool = [ MProcess(target=UploaderFunction, args=(params_dict,mp_i)) for mp_i in range(len(args.ctx))]
    for mp_i in mp_pool:
        mp_i.start()
    ###

    model = get_segmentation_model(model=args.model, dataset=args.dataset,
                                           backbone=args.backbone, norm_layer=args.norm_layer,
                                           norm_kwargs=args.norm_kwargs, aux=args.aux,
                                           base_size=args.base_size, crop_size=args.crop_size, ctx=args.ctx)
    model.cast(args.dtype)


    model.load_parameters(args.params_file,ctx=args.ctx);

    model_ev = DataParallelModel(model, args.ctx)

    args_strip = args.strip_corner_pixels;  # crop pixels
    for ev_input, (ev_rawpng, ev_coords) in eval_data:
        
        results = model_ev(ev_input);

        ev_input  = mx.gluon.utils.split_data(ev_input ,len(args.ctx),even_split=False)
        ev_rawpng = mx.gluon.utils.split_data(ev_rawpng ,len(args.ctx),even_split=False)
        ev_coords = mx.gluon.utils.split_data(ev_coords ,len(args.ctx),even_split=False)

        results = [ x[0] for x in results ];
        result_argmax = [ x.argmax(1).astype('uint8').copyto(mx.cpu()) for x in results]

        mx.nd.waitall();

        for r_a in range(len(ev_input)):
          for r_b  in range(len(ev_input[r_a])):
             if (ev_coords[r_a][r_b][0] == -1):
               continue; # skipping empty
             item = {};

             item['area'] = args.wkt_area_id;
             item['coords'] = ev_coords[r_a][r_b].asnumpy();
             if (args_strip > 0):
               item['result'] = result_argmax[r_a][r_b].asnumpy()[args_strip:-args_strip,args_strip:-args_strip];
               item['coords']+= np.array((math.floor(args_strip/pixel_scale),math.floor(args_strip/pixel_scale),-2*args_strip,-2*args_strip))
             else:
               item['result'] = result_argmax[r_a][r_b].asnumpy()

             mp_q.put(item);


        if (rank == 0):
          s = MPI.Status();
          while comm.Iprobe(status=s,tag=UPDATE_TAG):
               update_msg = comm.recv(tag=UPDATE_TAG)
               tbar.update(update_msg[0]);
          tbar.update(1)
        else:
          comm.isend(np.array([1],dtype='int'),dest=0,tag=UPDATE_TAG)



    if (rank == 0):
      print("Waiting for other nodes to finish");

    mp_e.set();
    for mp_i in mp_pool:
        mp_i.join()

    comm.Barrier()
