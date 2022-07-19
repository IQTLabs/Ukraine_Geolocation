#!/usr/bin/env python

import argparse
import subprocess
from PIL import Image
import time
import tqdm

from scenic import *

def web_features(input_file, output_file, temp_dir, view, arch, suffix, num, backup_interval, pause, key):
    """
    Given a dataset file with latitude/longitude, generate
    feature vectors using imagery from an API.
    """
    # Get key value from key path
    with open(key, 'r') as keyfile:
        key = keyfile.read().rstrip()

    # Load model and dataset
    transform = get_transform(view, preprocess=False, finalprocess=True)
    model = load_model(view, arch, suffix).to(device)
    dataset = OneDataset(input_file, view)
    if num > 0:
        dataset.df = dataset.df[:num]
        dataset.paths_relative = dataset.paths_relative[:num]
    dataset.df = pd.concat([dataset.df.iloc[:, :3], pd.DataFrame(np.zeros((len(dataset.df), 365), dtype=np.float32))], axis=1)

    # Loop through geotags
    for i in tqdm.tqdm(range(len(dataset))):
        lat = dataset.df.iloc[i]['lat']
        lon = dataset.df.iloc[i]['lon']
        filepath = os.path.join(temp_dir, 'image_' + lat + '_' + lon + '.jpg')

        # Download image
        if not os.path.exists(filepath):
            if pause > 0:
                time.sleep(pause)
            cmd = 'wget -O ' + filepath + ' "https://dev.virtualearth.net/REST/v1/Imagery/Map/Aerial/' + lat + ',' + lon + '/18?mapSize=800,800&key=' + key + '"'
            subprocess.check_output(cmd, shell=True)

        # Extract feature vector
        image = Image.open(filepath)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = transform(image).to(device)
        featvec = model(image.unsqueeze(0)).squeeze()
        dataset.df.iloc[i, 3:] = featvec.cpu().detach().numpy()

        # Delete image
        cmd = 'rm -v ' + filepath + ' 1>&2'
        subprocess.check_output(cmd, shell=True)

        # Backup feature vectors at regular intervals
        if (i + 1) % backup_interval == 0:
            backup_file = os.path.splitext(os.path.basename(output_file))[0] \
                          + '_backup' + str(i+1) \
                          + os.path.splitext(os.path.basename(output_file))[1]
            backup_path = os.path.join(temp_dir, backup_file)
            dataset.save(backup_path)

    # Save output
    dataset.save(output_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input',
                        default='../points/points_formatted.txt',
                        help='Path to input dataset file, with lat/lon')
    parser.add_argument('-o', '--output',
                        default='../points/points_features.txt',
                        help='Path to output dataset file, with features')
    parser.add_argument('-t', '--tempdir',
                        default='../points/temp',
                        help='Path to directory for backup files')
    parser.add_argument('-v', '--view',
                        default='overhead',
                        help='View: surface or overhead')
    parser.add_argument('-a', '--arch',
                        default='alexnet',
                        help='Model architecture')
    parser.add_argument('-s', '--suffix',
                        default=None,
                        help='Suffix of model weights file name')
    parser.add_argument('-n', '--num',
                        type=int, default=0,
                        help='How many dataset point to use, or 0 for all')
    parser.add_argument('-b', '--backup',
                        type=int, default=1000,
                        help='Number of images between backups')
    parser.add_argument('-p', '--pause',
                        type=float, default=0.,
                        help='Pause between API calls')
    parser.add_argument('-k', '--key',
                        default='key',
                        help='Path to API key')
    args = parser.parse_args()
    web_features(args.input, args.output, args.tempdir, args.view, args.arch, args.suffix, args.num, args.backup, args.pause, args.key)
