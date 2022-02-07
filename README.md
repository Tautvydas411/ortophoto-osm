# ortophoto-osm
This is public code repository for Lithuania geospatial data segmentation . The code is used for investigation of neural networks using openstreetmap database as labeling source. More detail are in "Urban Change Detection from Aerial Images Using Convolutional Neural Networks and Transfer Learning" paper.

**note: due to sheer size of volume and data licensing the project have direct links in storage which need to be modified acordingly**

## Required data
* Ortophoto https://www.geoportal.lt/geoportal/en/web/en - ~12Tb raw data (it can be smaller after compression)
* OpenStreetmap https://planet.openstreetmap.org/ - for this model we used full country Lithuania, but for other data it necessary to filter by boundaries

## Project specifics
Project is mostly self contained and required following packages
* mxnet
* gluon-cv
* GDAL python packages
* diskcache
* expiringdict
* shapely
* rasterio
* pyproj
