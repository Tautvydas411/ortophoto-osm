# ortophoto-osm
This is public code repository for Lithuania geospatial data segmentation. It is used by Kaunas University of Technology scientists. The code is used for investigation of neural networks using openstreetmap database as labeling source. More detail are in "Urban Change Detection from Aerial Images Using Convolutional Neural Networks and Transfer Learning" paper.

**note: due to sheer size of volume and data licensing the project have direct links to storage, which need to be modified acordingly (including EPSG transformations functions)**

## Required data
* Ortophoto https://www.geoportal.lt/geoportal/en/web/en - ~12Tb raw data (it can be smaller after compression)
* OpenStreetmap https://planet.openstreetmap.org/ - for this model we used full country Lithuania, but for other data it necessary to filter by boundaries

## Project strucrure
* pictureslib - vector,raster image processing utilities and dataset iterator
* dataset_prepare - labelled dataset generator from openstreetmap data
* training - training scripts
* segmentation - inference scripts (warning: it uses private ceph s3 gateway directly)

As noted above, there are multiple places which s3 users and keys which needs to be changed.
The ortophoto data have static pathes which in practise is mosaic build by GDAL library (for example  /mnt/mosaic/period-1b.vrt )



## Install requiments
Project main requirements are:
* mxnet
* gluon-cv
* GDAL python packages
* diskcache
* expiringdict
* shapely
* rasterio
* pyproj
* mpi4py
* psycopg2

it can be installed in conda envirioment:
```
$ mamba install -c conda-forge mxnet gdal diskcache expiringdict shapely rasterio pyproj mpi4py psycopg2
$ pip install gluon-cv
```
## Licensing

The Data licence is different:
* Ortophoto is from  Spatial Information Portal of Lithuania -  https://www.geoportal.lt/geoportal/en/web/en 
* OpenStreetMap - OpenStreetMap®  https://www.openstreetmap.org/copyright
* The code by default is licenced as The 3-Clause BSD License unless is noticed differently

