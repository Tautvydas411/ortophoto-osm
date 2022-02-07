#!/bin/bash

# execute  segmentation command

PARAMS="finelearning-coarse-pretrained-p3-epoch_0044_mIoU_0.8254.params"

BASE="s3://scratch-bucket-20211220/r1-"

#PERIODS=("x" "I" "II" "III")
PERIODS=("x" "III" "x" "x")

for p in {1..3};
do
  echo "Processing  ${PERIODS[$p]} period"

  for num in {40..45};
  do
	  ubase=$BASE${PERIODS[$p]}

	  CMD="python3 ./cloudsegmentation.py --period ${PERIODS[$p]} --params_file $PARAMS --s3-base-url $ubase --wkt-areas-file savivaldybes.wkt --wkt-area-id $num --workers 6 --strip-corner-pixels 64"
	  $CMD

  done
done
