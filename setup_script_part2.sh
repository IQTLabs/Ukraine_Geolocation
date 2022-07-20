#!/usr/bin/env bash

# Extracting CrossView USA
cd data
tar -xf flickr.tar
tar -xf flickr_aerial.tar
tar -xf streetview.tar
tar -xf streetview_aerial.tar
tar -xf metadata.tar

# File list
cat flickr_images.txt streetview_images.txt > all_images.txt

# Preprocessing
cd ../code
conda activate geoloc
python -c "from scenes import *; preprocess('../data/all_images.txt', '../data/preprocessed', view='surface')"

# Reorganizing folders
cd ../data
mv flickr flickr_orig
mv streetview streetview_orig
mv preprocessed/flickr flickr
mv preprocessed/streetview streetview
ln -s flickr_aerial flickr_aerial_full
ln -s streetview_aerial streetview_aerial_full
rmdir preprocessed
