#!/usr/bin/env bash

# Folders
mkdir data
mkdir weights

# Places2 weights
wget http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar -O weights/alexnet_surface.pth.tar
wget http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar -O weights/densenet161_surface.pth.tar

# Code
git clone https://github.com/IQTLabs/Ukraine_Geolocation.git code
cd code
wget https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt

# Python environment
conda create --name geoloc --file requirements.txt
