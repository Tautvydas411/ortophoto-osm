#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 26 01:19:18 2019

@author: as
@version 1.1v

v1.1 - Changed input format

"""


from diskcache import FanoutCache
from expiringdict import ExpiringDict
import imageio; #!conda install imageio
from expiringdict import ExpiringDict


import numpy as np;
import boto3;
import io;
import PIL.Image;
import random;

import mxnet as mx;
import gluoncv as gcv;
import sys;

import pickle;
import time;
import os.path

TESTING=True

try:
    import zstd; #!pip install zstd
except Exception as e:
    if TESTING == False:
        raise e; 
    import imp;
        
    my_module = imp.new_module('zstd')
    sys.modules['zstd'] = my_module;    
    my_module.ZSTD_compress = lambda x: x;
    my_module.ZSTD_uncompress = lambda x: x;
    print("!!!!!!!!!WARNING!!!!!!!!!!!!! zstd noutfound")
    import zstd;
    pass

import yaml; #!conda install pyaml (lyg ir sitas)

imageio.plugins.freeimage.download();


if (os.path.isfile(".fastcache")):
  print(".fastcache file is found in working directory, using FanoutCache for storage");
  cache_train = FanoutCache(directory="TrainCache",shards=16,timeout=5,size_limit=80*1024*1024*1024);
  cache_test = FanoutCache(directory="TestCache",shards=16,timeout=5,size_limit=16*1024*1024*1024);
else:
  print(".fastcache file isnot  found in working directory, using ExpiringDict for temporary storage");
  cache_train = ExpiringDict(max_len=16, max_age_seconds=3600);
  cache_test  = ExpiringDict(max_len=16, max_age_seconds=3600);


print("!!!Warning check caches!!!");


class CloudDataSet(mx.gluon.data.Dataset):

    __slots__ = ["current_cache","s3_client"];
    
    classes = ['Namas'];
    
    def getRepPNGFromTXT(self,txtkey,s3_client,BUCKET,extend=16):
       while True:
         try:
           return self.getPNGFromTXT(txtkey,s3_client,BUCKET,extend);
         except Exception as e:
           print("Error fetching %s, retrying in 5s (excepion %s)" % (txtkey,e));
           time.sleep(5);


    def getPNGFromTXT(self,txtkey,s3_client,BUCKET,extend=16):
     """
     example: flat/2009-2658888-7343647.txt
     """
     while True:
         #print("Getting '%s'" % txtkey);
         self.CheckClient();
         bytes_buffer = io.BytesIO()
         self.s3_client.download_fileobj(Bucket=BUCKET,Key=txtkey[:-9]+'-rawf.png',Fileobj=bytes_buffer)
         bytes_buffer.seek(0);
         img =  imageio.imread(bytes_buffer,'png-fi');
         
         bytes_buffer.seek(0);
         
         #f1 = open('/tmp/outbuf','wb');
         #f1.write(bytes_buffer.read());
         #f1.close();
         
         bytes_buffer = io.BytesIO()
         self.s3_client.download_fileobj(Bucket=BUCKET,Key=txtkey,Fileobj=bytes_buffer)
         bytes_buffer.seek(0); 
         bboxes = np.genfromtxt(bytes_buffer,dtype='float32');
         
         bytes_buffer = io.BytesIO()
         self.s3_client.download_fileobj(Bucket=BUCKET,Key=txtkey[:-9]+'-mask.png',Fileobj=bytes_buffer)
         bytes_buffer.seek(0);
         mask_img =PIL.Image.open(bytes_buffer);
         
         
         if (len(bboxes) == 0):     # pick other random item
             txtkey = random.sample(self._data,1)[0];
             continue;
         elif (len(bboxes.shape) == 1):
             n = np.zeros((1,bboxes.shape[0]+2),dtype='float32')
             n[0,0:bboxes.shape[0]] = bboxes;
         else:
             n = np.zeros((bboxes.shape[0],bboxes.shape[1]+2),dtype='float32');
             n[0:bboxes.shape[0],0:bboxes.shape[1]] = bboxes;
         
         
         n[:,:2] -=extend;
         n[:,2:4] +=extend;
         n[ n < 0] = 0;
         n[ n > 1024] = 1024;
         return (np.array(img),n,np.array(mask_img));
 
    def __init__(self,datadict : dict, the_test_set : bool = False, s3bucket : str =  'ortofo20092018-prepared', num_shards = 1, shard = 0):
        super(CloudDataSet,self).__init__();
        
        self.the_test_set = the_test_set;
           
        if (self.the_test_set):
            global cache_test;
            self.current_cache = cache_test;
        else:
            global cache_train;
            self.current_cache = cache_train;

        self._data = datadict[ "testset" if the_test_set else "trainset"];



        # taken from python/mxnet/gluon/data/dataset.py shard function
        assert shard < num_shards, 'Shard index of out bound: %d out of %d'%(shard, num_shards)
        assert num_shards > 0, 'Number of shards must be greater than 0'
        assert shard  >= 0, 'Index must be non-negative'

        length = len(self._data)
        shard_len = length // num_shards
        rest = length % num_shards
        # Compute the start index for this partition
        start = shard_len * shard + min(shard, rest)
        # Compute the end index for this partition
        end = start + shard_len + (shard < rest)

        self._data = self._data[start:end];

        self.BUCKET = s3bucket;

        if ('s3bucket' in datadict):
            if ('s3buckettest' in datadict and the_test_set == True):
              self.BUCKET = datadict['s3buckettest'];
            else:
              self.BUCKET = datadict['s3bucket'];

        self.s3_client = None;
    
    def __getstate__(self):
        state = self.__dict__.copy();
        if 's3_client' in state:    del state['s3_client'];
        if 'current_cache' in state:del state['current_cache'];
        return state;
        
    def __setstate__(self, state):
        self.__dict__.update(state)
        self.s3_client = None;
        if (self.the_test_set):
            global cache_test;
            self.current_cache = cache_test;
        else:
            global cache_train;
            self.current_cache = cache_train;

       
    def CheckClient(self):
        if (self.s3_client is None):
            session = boto3.session.Session()
            self.s3_client = session.client(
                service_name='s3',
                aws_access_key_id='xxxx',
                aws_secret_access_key='xxxx',

                endpoint_url='https://s3.litnet.lt',
                )


    def __len__(self):
        return len(self._data);
    
    def __getitem__(self, idx):
        
        the_key = self._data[idx];
        if ( the_key in self.current_cache):
            a,b,c = pickle.loads(zstd.ZSTD_uncompress(self.current_cache[the_key])); 
        else:
            a,b,c = self.getRepPNGFromTXT(self._data[idx],self.s3_client,self.BUCKET);
            
            a = a.astype('float32');
            
            p1 = np.percentile(a,2)
            p2 = np.percentile(a,98);
            
            a[ a < p1] = p1;
            a[ a > p2] = p2;
            
            a = a - p1;
            if (p2 -p1 ) == 0:
                a = a * 0;
            else:
                a = a / (p2 -p1);
            
            self.current_cache[the_key] = zstd.ZSTD_compress(pickle.dumps((a,b,c)));
        return (mx.nd.array(a,dtype='float32'),b);


class CloudSegmentationDataSet(CloudDataSet):
        
    
    classes = ['background','house','forest','water','grass','pavement']
    
    NUM_CLASS = 6
    
    @property
    def num_class(self):
      return 6;

    @property
    def pred_offset(self):
        return 0


    def __init__(self,datadict : dict, the_test_set : bool = False, transform = None):
        super(CloudSegmentationDataSet,self).__init__(datadict,the_test_set);
        self.transform = transform;
        
    def __getitem__(self, idx):
        
        the_file = self._data[idx // 3];
        the_key = "%s" % idx;
        the_period = (idx % 3)+ 1;
        
        if ( the_key in self.current_cache):
            a,b,c = pickle.loads(zstd.ZSTD_uncompress(self.current_cache[the_key])); 
        else:
            a,b,c = self.getRePNGSegmentation(the_file,the_period,self.s3_client,self.BUCKET);
            
            a = a.astype('float32');
            
            p1 = np.percentile(a,2)
            p2 = np.percentile(a,98);
            
            a[ a < p1] = p1;
            a[ a > p2] = p2;
            
            a = a - p1;
            if (p2 -p1 ) == 0:
                a = a * 0;
            else:
                a = a / (p2 -p1);
            
            self.current_cache[the_key] = zstd.ZSTD_compress(pickle.dumps((a,b)));
        return (self.transform(mx.nd.array(a,dtype='float32')),b);

class CloudSegmentationDataSetV2(CloudDataSet):
        
    
    #classes = ['background','house','forest','water']
    CLASSES = ['background','house','forest','water']

    NUM_CLASS = 4

    @property
    def num_class(self):
      return self.class_nr;

    @property
    def pred_offset(self):
        return 0


    def getPNGSegmentation(self,txtkey,period,s3_client,BUCKET,extend=16):
     """
     example: flat/2009-2658888-7343647.txt
     """
     while True:
         #print("Getting '%s'" % txtkey);
         self.CheckClient();
         
         
         
         maskfile = txtkey;
         imgfile = "p%d-orig-%s" % (period,maskfile.rsplit('-',1)[1]);
         
         imgfile = txtkey.rsplit('/',1)[0] + "/"+imgfile;
         #print(imgfile);
         bytes_buffer = io.BytesIO()
         self.s3_client.download_fileobj(Bucket=BUCKET,Key=imgfile,Fileobj=bytes_buffer)
         bytes_buffer.seek(0);
         img =  imageio.imread(bytes_buffer,'png-fi');
         
         bytes_buffer = io.BytesIO()
         self.s3_client.download_fileobj(Bucket=BUCKET,Key=maskfile,Fileobj=bytes_buffer)
         bytes_buffer.seek(0);
         mask_img =PIL.Image.open(bytes_buffer);
    

         
      
         return (np.array(img[:,:,:3]),np.array(mask_img));


    def getRePNGSegmentation(self,txtkey,s3_client,BUCKET,extend=16):
       while True:
         try:
           return self.getPNGSegmentation(txtkey,s3_client,BUCKET,extend);
         except Exception as e:
           print("Error fetching %s, retrying in 5s (excepion %s)" % (txtkey,e));
           time.sleep(5);


    def __init__(self,datadict : dict, the_test_set : bool = False, transform = None, s3bucket : str =  'ortofo20092018-prepared', num_shards =1, shard = 0, force_period = -1, force_class = -1):
        super(CloudSegmentationDataSetV2,self).__init__(datadict,the_test_set,s3bucket, num_shards = num_shards, shard = shard);
        self.transform = transform;
        assert not 'only_buildings' in datadict

        self.force_period = force_period
        self.force_class = force_class

        if (force_class == -1):
          self.class_nr = 4;
        else:
          self.class_nr = 2



    def __len__(self):
        if (self.force_period == -1):
          return len(self._data) * 3;
        else:
          return len(self._data);

    def __getitem__(self, idx):
        
        if (self.force_period == -1):
          the_file = self._data[idx // 3];
          the_key = "%s" % idx;
          the_period = (idx % 3)+ 1;
        else:
          the_file = self._data[idx];
          the_key = "%s" % idx;
          the_period = self.force_period;


        if ( the_key in self.current_cache):
            a,b = pickle.loads(zstd.ZSTD_uncompress(self.current_cache[the_key])); 
        else:
            a,b = self.getRePNGSegmentation(the_file,the_period,self.s3_client,self.BUCKET);
            
            a = a.astype('float32');
            
            p1 = np.percentile(a,2)
            p2 = np.percentile(a,98);
            
            a[ a < p1] = p1;
            a[ a > p2] = p2;
            
            a = a - p1;
            if (p2 -p1 ) == 0:
                a = a * 0;
            else:
                a = a / (p2 -p1);
            if (self.force_class != -1):  # if class is forced -> everything is background or object
              b[ b != self.force_class] = 0;
              b[ b == self.force_class] = 1

            self.current_cache[the_key] = zstd.ZSTD_compress(pickle.dumps((a,b)));
        return (self.transform(mx.nd.array(a,dtype='float32')),mx.ndarray.moveaxis(mx.nd.one_hot( mx.nd.array(b.astype('int') - self.pred_offset, mx.cpu(0)),self.num_class).astype('float32'),2,0),the_period);

    
if __name__ == '1__main__':
    print("Checking connection")

    testdata = """s3bucket: ortofo20092018-prepared
testset:
- good/v2/209166147-2733021-7295207/original_images/mask-209166147.png
- good/v2/584836251-2391446-7426539/original_images/mask-584836251.png
- good/v2/271812808-2462707-7434392/original_images/mask-271812808.png
- good/v2/310772474-2346137-7560419/original_images/mask-310772474.png
- good/v2/441692988-2782813-7201381/original_images/mask-441692988.png
trainset:
- good/v2/209166147-2733021-7295207/original_images/mask-209166147.png
- good/v2/584836251-2391446-7426539/original_images/mask-584836251.png
- good/v2/271812808-2462707-7434392/original_images/mask-271812808.png
- good/v2/310772474-2346137-7560419/original_images/mask-310772474.png
- good/v2/441692988-2782813-7201381/original_images/mask-441692988.png
"""

    ds = yaml.load(testdata, Loader=yaml.FullLoader);
    a = pickle.loads(pickle.dumps(CloudDataSet(ds)))
    b = pickle.loads(pickle.dumps(CloudDataSet(ds,the_test_set = True)));

    for i in range(4):
      r1 = a[i];
      r2 = b[i];
      print(r1[0].shape,r2[0].shape);
      #print(mx.nd.mean(r1[0]),mx.nd.mean(r2[0]))
    test2 ="""trainset:
- good/v1/288152785-2366346-7561813/original_images/mask-288152785.png
- good/v1/403099854-2811583-7287007/original_images/mask-403099854.png
- good/v1/508528524-2379871-7424503/original_images/mask-508528524.png
"""
    ds = yaml.load(test2, Loader=yaml.FullLoader);
    a = pickle.loads(pickle.dumps(CloudSegmentationDataSetV2(ds)))
    a.transform = lambda x: x;
    a[0];
