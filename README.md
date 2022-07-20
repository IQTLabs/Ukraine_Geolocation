# Ukraine Geolocation

The code in this repo uses deep learning to geolocate photographs (such as from social media) by comparison to satellite imagery.  The same trained model can be used for neighborhood-scale or national-scale geolocation.  This code is used in [Deep Geolocation with Satellite Imagery of Ukraine](https://www.iqt.org/blog/).

## Setup & Training

Download the two setup scripts, then run the first setup script:
```
./setup_script_part1.sh
```

Download CVUSA tar files into the "data" folder.  To get these files, follow instructions [here](https://mvrl.cse.wustl.edu/datasets/cvusa/).

Run the second setup script:
```
./setup_script_part2.sh
```

Train the model:
```
# In Python (run within code/ folder)
from scenes import *
extract_features('../data/all_images.txt',
                 '../data/all_surface.pkl',
                 view='surface', rule='cvusa',
                 populate_latlon=True)
train('../data/all_surface.pkl',
      view='overhead', rule='cvusa', arch='densenet161', batch_size=128)
```
Press Ctrl+c to end training.

## Neighborhood-Scale Geolocation

To see all syntax options:
```
./heatmap.py -h
```

### Usage Example

Save a file called `photo_paths.csv`, containing paths to ground-level photographs, one per line.  Let `LINE_NUM` be the line with the photo you want to use.  Then in the command below, change the projection (here, UTM 36N) and bounds (here, centered on Bucha/Irpin) as needed.  Note: This is an example only; the repo does not include photos or satellite imagery.
```
./heatmap.py --projection 32636 --bounds 300211 5598433 306475 5604575 --satpath SATELLITE_PHOTO.tif --photopath photo_paths.csv --csvpath OUTPUT_FILE_NAME.csv --row LINE_NUM --match --arch densenet161 --offset 10
```

## National-Scale Geolocation

To see all syntax options:
```
./websample.py -h
```

### Usage Example

Run `roadpoints` to get a list of random geographic points, then run `websample` to get a feature vector for each point, then use the `score_file` Python function to compare a photo to those feature vectors, as shown below.
Note: This requires your Bing Maps API key saved to a file at `PATH_TO_KEY`.
```
./roadpoints.py --output points.csv
./websample.py --input points.csv --output features.csv --tempdir TEMPORARY_DIRECTORY --arch densenet161 --pause 10 --num 10000 -k PATH_TO_KEY
```
Letting `photo_paths.csv` and `LINE_NUM` have the same meaning as in the previous usage example:
```
# In Python: (run within code/ folder)
from scenes import *
score_file('features.csv',
           'scores.csv',
           'photo_paths.csv', LINE_NUM)
```
The file `scores.csv` will show how well each point matches the selected photo.  In this file, 2nd column is latitude, 3rd column is longitude, and 4th column is score.  Score is the negative of the feature space distance.

## Credits

`scenes.py` is derived from [`cvig_fov.py`](https://github.com/IQTLabs/WITW/blob/main/model/cvig_fov.py).  `heatmap.py` is derived from [`heatmap.py`](https://github.com/IQTLabs/WITW/blob/main/tools/heatmap/heatmap.py).  The files in this repo are released under the same license (Apache 2.0) as those source files.
