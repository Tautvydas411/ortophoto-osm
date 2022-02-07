#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 13:30:53 2019

@author: as
Sckriptas skirtinti isrinkti atsitiktinius namus
V1.1 Perdaryta pagal 4 pozymius
"""

import shapely;
import sys,os, gc;
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__))+"/../")
from pictureslib import orotogis;
from pictureslib import ortophoto;
import imageio; #!conda install imageio
import os;
import numpy as np;
import PIL.Image;
import cv2; # !pip install opencv-python
import time

import binascii;

imageio.plugins.freeimage.download();

NUM_RAND=2*2508; # Namu kiekis
ITER_LIMIT=5 # Iteracijos duomenu kiekis

SAVE_DATA=True

BASEDIR="data";

BLEND_ALPHA=0.6


CHUNKS_PER_PERIOD=44

print("Picking %d random houses " % NUM_RAND);



PERIODS_VRTS = ['/mnt/mosaic/period-1.vrt','/mnt/mosaic/period-2.vrt','/mnt/mosaic/period-3.vrt']


def getPeriodFromVRT(fname):
    if (fname.endswith("/mnt/mosaic/period-1.vrt")):
        return "p1";
    if (fname.endswith("/mnt/mosaic/period-2.vrt")):
        return "p2";
    if (fname.endswith("/mnt/mosaic/period-3.vrt")):
        return "p3";
    


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]
        

def GetTestPoints(hp):
    rand_houses = {};

    
    rand_houses_keys = [726043160,100320011]

    # Get Actual houses
    c = hp.conn.cursor()

    s =  c.execute("select osm_id,way from planet_osm_polygon where osm_id = ANY(%s)",(rand_houses_keys,))
    f = c.fetchall()

    
    for i in f:
        rand_houses[i[0]] = (shapely.wkb.loads(i[1],hex=True),i[0]);
    
    return rand_houses;

def GetTotalRandomPoints(hp):
    # Random points
    c = hp.conn.cursor()
    s =  c.execute("select 'dummy',geom from ST_Dump(ST_GeneratePoints((select way from planet_osm_polygon where admin_level='2'),%s))",(NUM_RAND,))
    f = c.fetchall()
    
    
    rand_houses = {};

    nn = 7010000000
    for i in f:
        rand_houses[nn] = (shapely.wkb.loads(i[1],hex=True),nn);
        nn = nn + 1;
        
    return rand_houses;


def GetRandompointsByHouses(hp):
      
    rand_houses = {};

    # Get NUM_RAND houses (collecting random polygons)
    with hp.conn.cursor() as c:
        while len(rand_houses) < NUM_RAND:
            s =  c.execute("select osm_id from (select osm_id,building from planet_osm_polygon  tablesample SYSTEM (0.01)) p where building is not null limit %s;",(ITER_LIMIT,))
            f = c.fetchall()
            
            for i in f:
                i = i[0];
                if (i not in rand_houses):
                    rand_houses[i] = True;
                    if (len(rand_houses) == NUM_RAND): break;
    
    rand_houses_keys = list(rand_houses.keys());

    # Get Actual houses
    c = hp.conn.cursor()

    s =  c.execute("select osm_id,way from planet_osm_polygon where osm_id = ANY(%s)",(rand_houses_keys,))
    f = c.fetchall()
    

    
    for i in f:
        rand_houses[i[0]] = (shapely.wkb.loads(i[1],hex=True),i[0]);
        
    return rand_houses;
 
def GetRandomDistribution():
    """
    Picks random distribution
    
    returns dict in format:
        key: Image file
        value: list of points 
    """


    #hp = orotogis.HomeProcess(host='localhost',port=6432); # by default connecting to 158.129.15.153
    hp = orotogis.HomeProcess(host='/tmp/andrius',port=5432,database='lietuva',user='as'); # by default connecting to 158.129.15.153

    tiledir = ortophoto.TileDirectory();

    rand_houses = GetRandompointsByHouses(hp);
    #rand_houses = GetTestPoints(hp);

    #rand_houses = GetTotalRandomPoints(hp);
    

    houses_data = {};
    houses_data_chunked = [];


    for z in PERIODS_VRTS:
         houses_data[z] = [];
    
    for i in rand_houses:
        for z in PERIODS_VRTS:
            houses_data[z].append(rand_houses[i]);


    for i in houses_data:
        chks = chunks(houses_data[z],NUM_RAND // CHUNKS_PER_PERIOD);
        for j in chks:
            houses_data_chunked.append((i,j));
            


    return houses_data_chunked;

def FetchData_init():
   global hp,tiledir,tilecache
 
   hp = orotogis.HomeProcess(host='/tmp/andrius',port=5432,database='lietuva',user='as'); # by default connecting to 158.129.15.153

   tiledir = ortophoto.TileDirectory();

   tilecache = {}

def FetchData(item):
    
    print("*********** Fetching homes for data ***********");
    
    global hp, tiledir,tilecache

    #hp = orotogis.HomeProcess(host='localhost',port=6432); # by default connecting to 158.129.15.153

    i = item[0];
    items = item[1];
    
    #items = unique_data_files[i];
    year_period = getPeriodFromVRT(i);
    print("Processing %s -> %s with %d items" % (year_period,i,len(items)));

    if i in tilecache:
      t1 = tilecache[i]
    else:
      t1 = tiledir.open(i);
      tilecache[i] = t1

    t1extent = t1.GetRasterSize();
    for j in items:
        gc.collect()
        osm_id = j[1];
        j = j[0];
        cntr    = tuple(j.centroid.coords)[0];
        print("Centroidas",cntr)
        rng = np.random.uniform(-500,500,size=2);

        cntr = (cntr[0]+rng[0],cntr[1]+rng[1])
        pix_cntr= t1.Coord3857ToPixel(*cntr);

        print("Object center coords=%r in tile=%r extent=%r"%(cntr,pix_cntr,t1extent));
       
        pix_cntr = (int(pix_cntr[0]),int(pix_cntr[1]));
        
        
        centroid = 512;
        width=1024;
        height=1024;
        scale_factor = 1;
        
        if (t1.GetGeoTransform()[1] < 0.5):
            centroid = 1024;
            width=2048;
            height=2048;
            scale_factor = 1;
        
        
        # ignore outof bounds
        if ((pix_cntr[0]- centroid < 0)  or (pix_cntr[0] + centroid > t1extent[0]) or
            (pix_cntr[1]- centroid < 0) or  (pix_cntr[1] + centroid > t1extent[1])) :
                print("%s -> !!!!!!!Out of bounds!!!!!" % i);
                continue;
      
        
       
        if (SAVE_DATA):
            
            p1 = t1.GetRGBGeoBounds3857(pix_cntr[0] - centroid,pix_cntr[1] - centroid,width,height)
            
            allmask = hp.GetAresMaskv2(t1,p1,pix_cntr[0] - centroid,pix_cntr[1] - centroid,width,height);

            
            if (np.sum(allmask == 2) < (1024*1024 * 0.2)): continue;  # Warning vandens skaicius

            bf = hp.GetTiffBuldingsFeatures(t1,pix_cntr[0] - centroid,pix_cntr[1] - centroid,width,height,False);

            img = bf[0];

            allmask = allmask.T;
            mask_img = PIL.Image.fromarray(allmask,mode='P');
            mask_img.putpalette(orotogis.prefered_color_u8_flat);
            
            
            if (scale_factor == 0.5):
              mask_img = mask_img.resize((1024, 1024), PIL.Image.NEAREST)             
              img = cv2.resize(img, dsize=(1024, 1024), interpolation=cv2.INTER_CUBIC)
              
            
        
            # automatic normalization
            a = img.astype('float32');
            
            p1 = np.percentile(a,2)
            p2 = np.percentile(a,98);
            
            a[ a < p1] = p1;
            a[ a > p2] = p2;
            
            a = a - p1;
            if (p2-p1) != 0:
               a = a / (p2 -p1);
            else:
               a = a  *0.0;
            
            imgnorm = a;
            # end of automatic normalization
            
            dbase =  "%s/%d-%d-%d" % (BASEDIR,osm_id,cntr[0],cntr[1])
            
            os.makedirs("%s/original_images" % dbase,exist_ok = True);
            os.makedirs("%s/normalized_images" % dbase,exist_ok = True)
            
            
            imageio.imsave("%s/original_images/%s-orig-%d.png" % (dbase,year_period,osm_id),img,'PNG-FI')
            if (not os.path.exists("%s/mask-%d.png" % (dbase,osm_id))):
                mask_img.save("%s/mask-%d.png" % (dbase,osm_id),"PNG",optimize = True)

            if (not os.path.exists("%s/normalized_images/mask-%d.png" % (dbase,osm_id))):
                mask_img.save("%s/normalized_images/mask-%d.png" % (dbase,osm_id),"PNG",optimize = True)

            if (not os.path.exists("%s/original_images/mask-%d.png" % (dbase,osm_id))):
                mask_img.save("%s/original_images/mask-%d.png" % (dbase,osm_id),"PNG",optimize = True)


            imageio.imsave("%s/normalized_images/%s-norm-%d.png" % (dbase,year_period,osm_id),imgnorm,'PNG-FI')      

            pilimgnorm= PIL.Image.fromarray((imgnorm *255).astype('uint8'));
            s1 = PIL.Image.blend(pilimgnorm , mask_img.convert('RGB'), BLEND_ALPHA)
            
            s1.save("%s/%s-%s-with-transp.jpg" % (dbase,year_period,osm_id));
            
            with open("%s/info-%s.txt" % (dbase,year_period),"w") as f:
              f.write(j.wkt+"\n");
              f.write("%f %f\n" % ((rng[0],rng[1],)))
          
            
    return item;
    print("****");
    
if __name__ == '__main__':
    import multiprocessing as mp;
    unq = GetRandomDistribution();
    pool = mp.Pool(mp.cpu_count(), FetchData_init);
    results = pool.map(FetchData, unq)
    #print(list(map(FetchData, unq)));
    data = list(results)
    print("items %d" % len(data))
    print("Finishing pool")

    while len(pool._cache) > 0:
        time.sleep(0.2)

    print("data: %r" % unq)
    print("results: %r" % data) 
     
