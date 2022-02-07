#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 00:25:04 2019

@author: as
"""

from osgeo import gdal;
import numpy as np;

import weakref
from shapely import geometry,geos,ops,prepared
import pyproj;
from functools import partial
import rasterio;

import imageio;

geos.WKBWriter.defaults['include_srid'] = True


class OneTile:    
    """
    Wrapper class for tile
    """
    __slots__ = ['gd','__weakref__']
    gd: gdal.Dataset;


    _project3346To3857 = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3346'), # source coordinate system
    pyproj.Proj(init='epsg:3857')) # destination coordinate system

    _project3856To3346 = partial(
    pyproj.transform,
    pyproj.Proj(init='epsg:3857'), # source coordinate system
    pyproj.Proj(init='epsg:3346')) # destination coordinate system

    
    def __init__(self,uri):

        self.gd = gdal.Open(uri)
    def GetRGB(self,xoff,yoff,w,h):
        """
        Return image
        """
        bR = self.gd.GetRasterBand(1)
        bG = self.gd.GetRasterBand(2)
        bB = self.gd.GetRasterBand(3)
        
        r1 = bR.ReadAsArray(xoff,yoff,w,h).T;
        r2 = bG.ReadAsArray(xoff,yoff,w,h).T;
        r3 = bB.ReadAsArray(xoff,yoff,w,h).T;
        
        r123 = np.stack((r1,r2,r3)).T;

        
        return r123;
    
    def GetRGBGeoBounds(self,xoff,yoff,w,h) -> geometry.Polygon :
        """
        Get RGB geographic bounds in geographic shape
        @warning ingores shear component
        @warning SRID set to 3346
        """
        t = self.GetGeoTransform();
        
        minx = t[0] +(xoff)*t[1];
        maxx = t[0] +(xoff+w)*t[1];

        miny = t[3] +(yoff+h)*t[5];
        maxy = t[3] +(yoff)*t[5];
        
        box = geometry.box(minx,miny,maxx,maxy);
        
        geos.lgeos.GEOSSetSRID(box._geom, 3346)
        
       
        return box
       
    def GetRGBGeoBounds3857(self,xoff,yoff,w,h) -> geometry.Polygon :
        p0 = self.GetRGBGeoBounds(xoff, yoff, w, h);
        geos.lgeos.GEOSSetSRID(p0._geom, 3857);
        return self.TransformTo3857(p0);


    def TransformTo3857(self,p0 : geometry.Polygon) -> geometry.Polygon():
        g2 = ops.transform(self._project3346To3857, p0)  # apply projection
        geos.lgeos.GEOSSetSRID(g2._geom, 3857);
        return g2;
        
        
    def GetGeoTransform(self):
        return self.gd.GetGeoTransform();
        pass
    def GetRasterSize(self):
        return (self.gd.RasterXSize,self.gd.RasterYSize);
    
    def Coord3857ToPixel(self, xorg: float, yorg: float) -> (int,int):
        """
        Return pixels from 3857 corrdantes
    
        Parameters
        ----------
        xorg : float
            x coordinate .
        yorg : float
            y coordinate
    
        Returns
        -------
        (int,int)
            DESCRIPTION.
    
        """
        k = self._project3856To3346(xorg,yorg);
        
        gt = self.GetGeoTransform();
        a =  rasterio.Affine.translation(gt[0],gt[3]) * rasterio.Affine.scale(gt[1],gt[5]);
        ainv = ~a;
    
        r = (ainv * k);
    
        return r;

    def Coord3346ToPixel(self, xorg: float, yorg: float) -> (int,int):
        """
        Return pixels from 3857 corrdantes
    
        Parameters
        ----------
        xorg : float
            x coordinate .
        yorg : float
            y coordinate
    
        Returns
        -------
        (int,int)
            DESCRIPTION.
    
        """
        k = np.array([xorg,yorg]);
        
        gt = self.GetGeoTransform();
        a =  rasterio.Affine.translation(gt[0],gt[3]) * rasterio.Affine.scale(gt[1],gt[5]);
        ainv = ~a;
    
        r = (ainv * k);
    
        return r;



    def SavePNG(self,filename : str, xoff : int, yoff:int, picture : np.array, world_file: bool = True):
        
        if not filename.endswith('.png'):
            raise   Exception("File name must end with png");
                        
        imageio.imwrite(filename,picture,'png-fi');
            
        if (world_file):
            worldname = filename.rsplit('.',1)[0] + ".wld";

            t = self.GetGeoTransform();
            
            x1 = (t[1]*xoff) + t[0];
            y1 = (t[5]*yoff) + t[3];
            
            with open(worldname,"w") as wf:
                wf.write("%f\n" % t[1]); # pixel x size
                wf.write("%f\n" % 0.0);
                wf.write("%f\n" % 0.0);
                wf.write("%f\n" % t[5]); # pixel x size
                wf.write("%f\n" % x1); # pixel x size
                wf.write("%f\n" % y1); # pixel x size
        pass;
        
    @staticmethod
    def SavePNGStatic(filename : str, xpixel_pos : int, ypixel_pos:int, picture : np.array, geotransform : tuple, world_file: bool = True):
        """
        Static method of save png.
        @see https://stackoverflow.com/questions/52443906/pixel-array-position-to-lat-long-gdal-python
        """
        if not filename.endswith('.png'):
            raise   Exception("File name must end with png");
                        
        imageio.imwrite(filename,picture,'png-fi');
            
        if (world_file):
            worldname = filename.rsplit('.',1)[0] + ".wld";

            xoff, a, b, yoff, d, e = geotransform;
            
            xp = a * xpixel_pos + b * ypixel_pos + a * 0.5 + b * 0.5 + xoff
            yp = d * xpixel_pos + e * ypixel_pos + d * 0.5 + e * 0.5 + yoff
    

            with open(worldname,"w") as wf:
                wf.write("%f\n" % t[1]); # pixel x size
                wf.write("%f\n" % 0.0);
                wf.write("%f\n" % 0.0);
                wf.write("%f\n" % t[5]); # pixel x size
                wf.write("%f\n" % x1); # pixel x size
                wf.write("%f\n" % y1); # pixel x size
        pass;

    def CreateTiles(self, input_poly:geometry.Polygon,w:int,h:int,w_overlap:int,h_overlap:int) -> [(float,float,float,float)]:
        """
        Creates rectangle tiles for input_poly. Technically is a bounding box of polygin with all tiles which intersects
        
        Parameters
        ----------
        input_poly : geometry.Polygon
            input polygon
        w          : int
            width
        h          : height
            height
        w_overlap  : int
           width overlap
        h_overlap  : int
           height overlap
        """
        
        bbox = np.floor(np.array(input_poly.bounds)+np.array([-1,-1,1,1]));
        
        pg = prepared.prep(input_poly);
        
        wstep = w - w_overlap;
        hstep = h - h_overlap;
        
        (x0,y0) = self.Coord3346ToPixel(bbox[0],bbox[1]);        
        (x1,y1) = self.Coord3346ToPixel(bbox[2],bbox[3]);
        
        x = x0;
        y = y1;
        
        xmax = x1;
        ymax = y0;
        gentiles = [];
                    
        while (y < ymax):
            while (x < xmax):
                
                
                gb = self.GetRGBGeoBounds(x,y,w,h);
                
                if (pg.intersects(gb)):
                    gentiles.append((x,y,w,h,gb));
                
                x +=wstep;
                
            y +=hstep;
            x = x0;
                                
        return gentiles;
class TileDirectory:
    """
    Tile directory for my server
    """

    __slots__ = ['refs']

    PREFIX='/mnt/mosaic';
    def __init__(self):
        gdal.SetConfigOption('GDAL_HTTP_UNSAFESSL', 'TRUE')
        gdal.SetConfigOption('AWS_S3_ENDPOINT','s3.litnet.lt');
        gdal.SetConfigOption('AWS_SECRET_ACCESS_KEY','');
        gdal.SetConfigOption('AWS_ACCESS_KEY_ID','');
        gdal.SetConfigOption('GDAL_HTTP_MAX_RETRY','99');
        gdal.SetConfigOption('AWS_VIRTUAL_HOSTING', 'FALSE');
        
        self.refs = weakref.WeakValueDictionary();
        pass

    def open(self,fname : str) -> OneTile:
        """
        Open raster from server
        """
        uri = '';
        if (fname.startswith('/mnt/mosaic/')):
            uri = self.PREFIX + fname[11:];
        else:
            raise Exception('only absolute path in form /mnt/mosaic is implemented')
        
        tile = None;
        if (uri in self.refs):
            tile = self.refs[uri];
            if (tile is not None):
                return tile
        
        uri='/vsis3/ortofo20092018'+uri;
        g = OneTile(uri);        
        self.refs[uri] = g;
        return g;
        
if __name__ == '__main__':
    print("Testing ortophoto server endpoint");
    td =TileDirectory();
    
    o = td.open('/mnt/mosaic/c/ORT10LT/2018/TIFF/72_44.tif')
    print(o.GetGeoTransform());
    
    o = td.open('/mnt/mosaic/period-3.vrt');
    tpix=o.GetRasterSize()[0]*o.GetRasterSize()[1] / 10**12;
    print("transform %r mosaic size %r or %.3f TeraPixels" % (o.GetGeoTransform(),o.GetRasterSize(),tpix));
