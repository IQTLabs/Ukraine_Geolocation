#!/usr/bin/env python

import argparse
from osgeo import osr
from osgeo import gdal

from scenic import *


class TileDataset(torch.utils.data.Dataset):
    def __init__(self, source, windows, transform=None):
        self.source = source
        self.windows = windows
        self.transform = transform
    def __len__(self):
        return len(self.windows)
    def __getitem__(self, idx):
        mem_path = '/vsimem/tile%s.jpg' % str(idx)
        ds = gdal.Translate(mem_path, self.source, projWin=self.windows[idx])
        raw = ds.ReadAsArray()
        gdal.GetDriverByName('GTiff').Delete(mem_path)
        image = torch.from_numpy(raw.astype(np.float32)/255.)
        data = {'image':image}
        if self.transform is not None:
            data['image'] = self.transform(data['image'])
        return data


def sweep(sat_path, bounds, edge, offset,
          photo_path, photo_row, csv_path, match):

    # Compute center and window for each satellite tile
    center_eastings = []
    center_northings = []
    windows = []
    e2 = edge / 2.
    for easting in np.arange(bounds[0] - e2, bounds[2] - e2, offset):
        for northing in np.arange(bounds[3] + e2, bounds[1] + e2, -offset):
            center_eastings.append(easting + e2)
            center_northings.append(northing - e2)
            windows.append([easting, northing, easting + edge, northing - edge])

    # Load satellite strip
    sat_file = gdal.Open(sat_path)

    # Specify transformations
    surface_transform = get_transform('surface', preprocess=False, finalprocess=True, augment=False, already_tensor=False)
    overhead_transform = get_transform('overhead', preprocess=False, finalprocess=True, augment=False, already_tensor=True)

    # Load data
    surface_set = OneDataset(photo_path, view='surface', rule=None,
                             transform=surface_transform)
    overhead_set = TileDataset(sat_file, windows, overhead_transform)
    surface_batch = torch.unsqueeze(surface_set[photo_row]['image'],
                                    dim=0).to(device)
    overhead_loader = torch.utils.data.DataLoader(overhead_set, batch_size=64,
                                                  shuffle=False, num_workers=1)

    # Load the neural networks
    surface_model = load_model('surface').to(device)
    overhead_model = load_model('overhead').to(device)
    # if device_parallel and torch.cuda.device_count() > 1:
    #     surface_model = torch.nn.DataParallel(
    #         surface_model, device_ids=device_ids)
    #     overhead_model = torch.nn.DataParallel(
    #         overhead_model, device_ids=device_ids)
    torch.set_grad_enabled(False)

    # Surface photo's features
    surface_vectors = surface_model(surface_batch)
    surface_vector = torch.unsqueeze(surface_vectors[0], 0)

    # Overhead images' features
    feat_vecs = None
    for batch, data in enumerate(tqdm.tqdm(overhead_loader)):
        images = data['image'].to(device)
        feat_vecs_part = overhead_model(images)
        if feat_vecs is None:
            feat_vecs = feat_vecs_part
        else:
            feat_vecs = torch.cat((feat_vecs, feat_vecs_part), dim=0)

    # Calculate score for each overhead image
    distances = torch.pow(torch.sum(torch.pow(feat_vecs - surface_vector, 2), dim=1), 0.5)

    # Find best scene match for each overhead image
    if match:
        scene_path = 'categories_places365.txt'
        scene_list = pd.read_csv(scene_path, sep=' ', header=None,
                         names=['scenes'], usecols=[0])['scenes'].tolist()
        match_indices = feat_vecs.argmax(dim=1).cpu().numpy()
        match_names = [scene_list[i] for i in match_indices]

    # Save information to disk
    df = pd.DataFrame({
        'x': center_eastings,
        'y': center_northings,
        'dissimilarity': distances.cpu().numpy(),
    })
    if match:
        df['match'] = match_names
    df.to_csv(csv_path, index=False)


def layer(sat_path, bounds, layer_path):
    sat_file = gdal.Open(sat_path)
    window = [bounds[0], bounds[3], bounds[2], bounds[1]]
    gdal.Translate(layer_path, sat_file, projWin=window)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--satpath',
                        default='satellite.tif',
                        help='Path to input satellite image')
    parser.add_argument('-b', '--bounds',
                        type=float,
                        nargs=4,
                        default=(305541, 5541833, 311833, 5548133),
                        metavar=('left', 'bottom', 'right', 'top'),
                        help='Bounds given as UTM coordinates in this order: min easting, min northing, max easting, max northing')
    parser.add_argument('-e', '--edge',
                        type=float,
                        default=368,
                        help='Edge length of satellite imagery tiles [m]')
    parser.add_argument('-o', '--offset',
                        type=float,
                        default=25,
                        help='Offset between centers of adjacent satellite imagery tiles [m]')
    parser.add_argument('-p', '--photopath',
                        default='./images.csv',
                        help='Path to surface photo dataset CSV file')
    parser.add_argument('-r', '--row',
                        type=int,
                        default=0,
                        help='Row of surface photo within CSV file')
    parser.add_argument('-c', '--csvpath',
                        default='./geomatch.csv',
                        help='Path to output CSV file path')
    parser.add_argument('-l', '--layerpath',
                        default='./satlayer.tif',
                        help='Path to output cropped satellite image')
    parser.add_argument('-i', '--image',
                        action='store_true',
                        help='Flag to output cropped satellite image')
    parser.add_argument('-g', '--gpu',
                        type=int,
                        default=None,
                        help='Which GPU to use')
    parser.add_argument('-m', '--match',
                        action='store_true',
                        help='Flag to include best scene matches in CSV file')
    args = parser.parse_args()
    if args.gpu is not None:
        cvig.device = torch.device('cuda:' + str(args.gpu))
        device = torch.device('cuda:' + str(args.gpu))
    sweep(args.satpath, args.bounds, args.edge, args.offset,
          args.photopath, args.row, args.csvpath, args.match)
    if args.image:
        layer(args.satpath, args.bounds, args.layerpath)