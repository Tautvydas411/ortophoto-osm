#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 22:48:25 2019

@author: as
Orotophoto gis lib
"""
import logging
from functools import partial
import psycopg2
from shapely import geometry, ops
import shapely.wkb
import rasterio.features # install via pip
import pyproj

from . import ortophoto

import numpy as np;


class HomeProcess:
    """
    Home raster processing class
    """
    conn : psycopg2.extensions.connection # pyopg connection
    
    _project3346To3857 = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3346'), # source coordinate system
    pyproj.Proj(init='epsg:3857')) # destination coordinate system

    _project3856To3346 = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3857'), # source coordinate system
    pyproj.Proj(init='epsg:3346')) # destination coordinate system


    def __init__(self, host = 'xxx.xxx', port = 5455, user = 'xxxx', password ='xxxx', database='xxx'):
        """
        Connect to SQL host for home processing and work with shapes
        """
        
        self.conn = psycopg2.connect(database=database,user=user,host=host,port=port,password=password)

        logging.info("PG host connection created to %r",self.conn)
        
    
    def GetAreaHousesShapes(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        """

        Parameters
        ----------
        area : geometry.Polygon
            area in which houses a located in SRID=3857

        Returns
        -------
        list of houses

        """
            
        c = self.conn.cursor()
        
        s = c.execute("select count(way) from planet_osm_polygon where building is not null and ST_Intersects(way,%s)",(area.wkb,));
        
        f = c.fetchall()[0];
        print("Num of building %d in %s" % (f[0],area.wkt))
        if (f[0] > 1000):
          print("High building count: %s", area.wkt);
          
        s =  c.execute("select way from planet_osm_polygon where building is not null and ST_Intersects(way,%s)",(area.wkb,))
        f = c.fetchall()
        
        lst = []
        
        for i in f:
            if (transformTo3346):
                pol = ops.transform(self._project3856To3346,shapely.wkb.loads(i[0],hex=True));
            else:
                pol  = shapely.wkb.loads(i[0],hex=True);
            lst.append(pol)   
         
            
        return lst;
 
    def GetTiffBuldingsFeatures(self,tile: ortophoto.OneTile, xoff : int, yoff : int,w : int,h : int,rect_sizes : bool = False,pad: int = 0):
            """
            Get Tiff region with houses masks

            Parameters
            ----------
            tile : ortophoto.OneTile
                Image tile.
            xoff : int
                x offset - in pixels
            yoff : int
                y offset - in pixels
            w : int
                width
            h : int
                height
            rect_sizes : bool
                transforms shape on bounding boxes
            pad : int
                add padding to shape
            Returns
            -------
            np.array(w,h,3) - image tile
            [geometry.Polygon] - houses polygon
            np.array(w,h) - mask of houses

            """
            
            p1 = tile.GetRGBGeoBounds3857(xoff, yoff, w, h)
            p1o = tile.GetRGBGeoBounds(xoff, yoff, w, h)
            
            gt = tile.GetGeoTransform()
            a =  rasterio.Affine.translation(p1o.bounds[0],p1o.bounds[3]) * rasterio.Affine.scale(gt[1],gt[5]);
            ainv = ~a;
            
            r = tile.GetRGB(xoff, yoff, w,h)


            l = self.GetAreaHousesShapes(p1,True);
            
            if (rect_sizes):
                lnew = [];
                for i in l:
                    b1 = geometry.box(*i.bounds);
                    lnew.append(b1);
                l = lnew;
            
            if (len(l) == 0):
                shp = np.zeros((h,w),dtype='uint8');
            else:
                shp = rasterio.features.rasterize(l,out_shape=(h,w),transform=a,all_touched=True);
                
            shp = shp.T;
        
            # Do translation for image coordinates (pixels):
            img_shapes = [];
            def microt(x,y):
                r = (ainv * (x,y));
                return (r[1],r[0])
            for i in l:
                img_shapes.append(ops.transform(microt, i));
            limagecoord = img_shapes;
            lcoord = l;
            return (r,shp,limagecoord,lcoord);
    
    
    def GetShapeMask(self, tile: ortophoto.OneTile, xoff : int, yoff : int,w : int,h : int,rect_sizes : bool = False,pad: int = 0):
        c = self.conn.cursor()
        
        area = tile.GetRGBGeoBounds3857(xoff, yoff, w, h)

        s =  c.execute("""select way from planet_osm_polygon where (landuse = 'forest' or "natural" = 'wood')  and ST_Intersects(way,%s)""",
                       (area.wkb,))
        f = c.fetchall()
        
        lst = []
        
        for i in f:
            if (transformTo3346):
                pol = ops.transform(self._project3856To3346,shapely.wkb.loads(i[0],hex=True));
            else:
                pol  = shapely.wkb.loads(i[0],hex=True);
            lst.append(pol)   

        pass
    
    def PostProcessImage(self, img :np.array, local_shp :[geometry.Polygon], neutral_img : np.array = None):
        """
            Post process image a.k.a remove out of bounds shapes
            It is three step method:
                1) Collect all shapes with pixels out of bounds of image
                2) Collect shapes which are definitely in image
                3) Substract inside shapes from outside (due to bounding box there will be overlaps)

        Parameters
        ----------
        img : np.array
            input array where conversion is done in plae
        local_shp : [geometry.Polygon]
            shapes in image pixels coordinates
        neutral_img : np.array
            neural image (size must be same as input image).
            if None, the median pixel value will be used

        Returns
        -------
        lst_inside - inside  shapes lists
        lst_outisde - outside shape lists

        mask_inside - mask inside picture
        """
        s = img.shape;
        
        if (neutral_img is  None):
            pix = np.array( (np.median(img[:,:,0]),np.median(img[:,:,1]),np.median(img[:,:,2])))
            neutral_img = np.zeros_like(img);
            neutral_img[:,:] = pix;
            print("Median",pix)
        
        lst_outside=[];
        lst_inside =[];
        
        for i in local_shp:
            b = np.array(i.bounds);
            bx = b[[0,2]]
            by = b[[1,3]]
            if (np.any(by < 0) or np.any(by > s[0])) or (np.any(bx < 0) or np.any(bx > s[1])):
                lst_outside.append(i);
            else:
                lst_inside.append(i);
        
        lst_outside = shapely.ops.unary_union(lst_outside);
        lst_inside = shapely.ops.unary_union(lst_inside);
        
        aa = lst_outside.difference(lst_inside)
        
        if  type(aa) == geometry.collection.GeometryCollection and  len(aa) == 0:
            shp = np.zeros(s[:2],dtype='uint8');
        else:
            shp = rasterio.features.rasterize([aa],out_shape=s[:2]);

        img[:,:,0][shp == 1] = neutral_img[:,:,0][shp == 1];
        img[:,:,1][shp == 1] = neutral_img[:,:,1][shp == 1];
        img[:,:,2][shp == 1] = neutral_img[:,:,2][shp == 1];
    
        if type(lst_inside) == geometry.collection.GeometryCollection and len(lst_inside) == 0:
            mask_inside =np.zeros(s[:2],dtype='uint8');
        else:
            mask_inside = rasterio.features.rasterize([lst_inside],out_shape=s[:2]);

    
        return (lst_inside,lst_outside,mask_inside)
  
    
    def GetAreaSQL(self, area: geometry.Polygon, sql_cond : str,transformTo3346 : bool = False) -> [geometry.Polygon]:
        c = self.conn.cursor()
        
        s =  c.execute("""select way from planet_osm_polygon where """+sql_cond +""" and ST_Intersects(way,%s)""",(area.wkb,))
        f = c.fetchall()
        
        lst = []
        
        for i in f:
            if (transformTo3346):
                pol = ops.transform(self._project3856To3346,shapely.wkb.loads(i[0],hex=True));
            else:
                pol  = shapely.wkb.loads(i[0],hex=True);
            lst.append(pol)   
         
            
        return lst;    

    def GetAreaBuildings(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        cond=""" (building is not null) """;
        return self.GetAreaSQL(area,cond,transformTo3346);

    def GetAreaForest(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        return self.GetAreaSQL(area,"""(landuse = 'forest' or "natural" = 'wood' )""",transformTo3346)    

    def GetAreaWater(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        cond="""(water is not null or waterway is not null or "natural"='water' or landuse='basin' or landuse='reservoir')""";
        return self.GetAreaSQL(area,cond,transformTo3346);

    def GetAreaGrass(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        cond="""("natural"='heath' or landuse='farmyard' or landuse='farmland' or landuse='meadow' or landuse ='allotments' or landuse='grass' or landuse='greenfield' or landuse='plant_nursery' or landuse='village_green')""";
        return self.GetAreaSQL(area,cond,transformTo3346);

    def GetAreaPavemently(self,area : geometry.Polygon, transformTo3346 : bool = False) -> [geometry.Polygon]:
        cond=""" (landuse='commercial' or landuse='industrial' or landuse='residential') """;
        return self.GetAreaSQL(area,cond,transformTo3346);
        
    def CreateAreaMask(self,tile: ortophoto.OneTile,  area_polys:[geometry.Polygon], xoff:int, yoff:int, w:int, h:int):
            #m1=hp.CreateAreaMask(t1,hp.GetAreaForest(p1,True),4000,4000,1024,1024)
        
            p1 = tile.GetRGBGeoBounds3857(xoff, yoff, w, h)
            p1o = tile.GetRGBGeoBounds(xoff, yoff, w, h)
            
            gt = tile.GetGeoTransform()
            a =  rasterio.Affine.translation(p1o.bounds[0],p1o.bounds[3]) * rasterio.Affine.scale(gt[1],gt[5]);
            ainv = ~a;

            if (len(area_polys) == 0):
                shp = np.zeros((h,w),dtype='uint8');
            else:    
                shp = rasterio.features.rasterize(area_polys,out_shape=(h,w),transform=a,all_touched=True);
                
            shp = shp.T;
            
            return shp;
       
        
    def GetAresMaskv2(self, tile: ortophoto.OneTile, p1:geometry.Polygon, xoff : int, yoff : int,w : int,h : int,rect_sizes : bool = False,pad: int = 0):
        """
        Create masks v2, create more practical masks:
            
        
        """

        area = np.zeros((w,h),dtype='uint8');

        buildings= self.CreateAreaMask(tile,self.GetAreaBuildings(p1,True),xoff,yoff,w,h) 
        forest   = self.CreateAreaMask(tile,self.GetAreaForest(p1,True),xoff,yoff,w,h)
        water    = self.CreateAreaMask(tile,self.GetAreaWater(p1,True),xoff,yoff,w,h)
        #grass    = self.CreateAreaMask(tile,self.GetAreaGrass(p1,True),xoff,yoff,w,h)
        #pavement = self.CreateAreaMask(tile,self.GetAreaPavemently(p1,True),xoff,yoff,w,h)
        
        area[water > 0] = 3;        
        area[forest > 0] = 2;
        area[buildings > 0] = 1;
        
        #area[grass > 0] = 0;
        #area[pavement > 0] = 0;
        
        return area;


    def GetAresMask(self, tile: ortophoto.OneTile, p1:geometry.Polygon, xoff : int, yoff : int,w : int,h : int,rect_sizes : bool = False,pad: int = 0):
        """
        Returns areas: forest, water, grass, pavement
        
        """
        area = np.zeros((w,h),dtype='uint8');


        forest   = self.CreateAreaMask(tile,self.GetAreaForest(p1,True),xoff,yoff,w,h)
        water    = self.CreateAreaMask(tile,self.GetAreaWater(p1,True),xoff,yoff,w,h)
        grass    = self.CreateAreaMask(tile,self.GetAreaGrass(p1,True),xoff,yoff,w,h)
        pavement = self.CreateAreaMask(tile,self.GetAreaPavemently(p1,True),xoff,yoff,w,h)
    
        # forest[forest == 1] = 2;
        # water[water == 1] = 3;
        # grass[grass == 1] = 4;
        # pavement[pavement == 1] = 5;

        area[forest > 0] = 2;
        area[water > 0] = 3;
        area[grass > 0] = 4;
        area[pavement > 0] = 5;
        return area;

prefered_color_map = [
    (1,1,1), # Unknown, background
    (0.9058823529411765, 0.1607843137254902, 0.5411764705882353), # House
    (0.6509803921568628, 0.4627450980392157, 0.11372549019607843), # Forest
    (0.4588235294117647, 0.4392156862745098, 0.7019607843137254), # Water
    (0.4, 0.6509803921568628, 0.11764705882352941), # Grass
    (0.4, 0.4, 0.4), # pavemently
    ]

__prefered_color_flat = ((np.array(prefered_color_map)).flatten()*255).astype('uint8');

prefered_color_u8_flat = np.zeros(768,dtype='uint8');
prefered_color_u8_flat[:__prefered_color_flat.shape[0]] = __prefered_color_flat;

prefered_color_u8_flat_short = prefered_color_u8_flat[0:3*6];

# 1 = PIL.Image.fromarray(its,'P')
# z1.putpalette(prefered_color_map_u8_flat)
if __name__ == '__main__': # Testing purpose
    print("Do test")
    from matplotlib import pyplot as plt;
    import copy;
    
    #logging.basicConfig(filename='/tmp/log1.log',level=logging.INFO)
    
    td = ortophoto.TileDirectory();
    #t1 = td.open("/mnt/mosaic/c/ORT10LT/2018/TIFF/57_32.tif");
    #t1 =td.open('/mnt/mosaic/c/ORT10LT/2018/TIFF/59_36.tif');
    t1 = td.open('/mnt/mosaic/b/ORT10LT/2016/60_23.tif');
    xoff=7000;
    yoff=5000;
    p1 = t1.GetRGBGeoBounds3857(xoff, yoff, 1024, 1024)
    p1o = t1.GetRGBGeoBounds(xoff, yoff, 1024, 1024)
    #r = t1.GetRGB(4400, 4800, 1024,1024)

    hp = HomeProcess();
    
    bf = hp.GetTiffBuldingsFeatures(t1,xoff,yoff,1024,1024,False);
 
    my_cmap = copy.copy(plt.cm.get_cmap('gray')) # get a copy of the gray color map
    my_cmap.set_bad(alpha=0) # set how the colormap handles 'bad' values
    
    fig = plt.figure(1);

    img = bf[0];
    if (img.max() > 255):
        img = img / img.max();
    plt.imshow(img)
    for i in bf[2]:
        plt.plot(*i.exterior.xy,color='red')
    #plt.imshow(bf[1],alpha=0.5,cmap=my_cmap)
    
    plt.show();
    
    pi = hp.PostProcessImage(img,bf[2])

    plt.figure(2);
    
    plt.imshow(img)
    for i in pi[0]:
        plt.plot(*i.exterior.xy,color='red')
    for i in pi[1]:
        plt.plot(*i.exterior.xy,color='blue')
    
    plt.show();
    # **** XXXX **** XXXX ***
    plt.figure(3);
    
    house    = pi[2];
   
    
    allmask = hp.GetAresMask(t1,p1,xoff,yoff,1024,1024);
    allmask[house > 0] = 1;
    
    plt.show();