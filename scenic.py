#!/usr/bin/env python

import os
import math
import time
import tqdm
import torch
import torchvision
import numpy as np
import pandas as pd
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device_parallel = True
device_ids = None

class OneDataset(torch.utils.data.Dataset):
    """
    Dataset for a single view (surface or overhead)
    Input file format is:
    path[,latitude,longitude[,feature_vector_components]]
    where brackets denote optional entries
    """
    def __init__(self, input_file, view='surface', zoom=18, rule='cvusa', transform=None):
        self.input_file = input_file
        self.view = view # surface, overhead
        self.zoom = zoom # 18, 16, 14
        self.rule = rule # cvusa, literal
        self.transform = transform

        # Load entries from input file
        typedict = {0:'string', 1:'string', 2:'string'}
        for i in range(3, 3+365):
            typedict[i] = 'float32'
        self.df = pd.read_csv(self.input_file, header=None, dtype=typedict)
        self.df.rename(columns={0:'path', 1:'lat', 2:'lon'}, inplace=True)
        if 'lat' not in self.df:
            self.df['lat'] = None
        if 'lon' not in self.df:
            self.df['lon'] = None

        # Create series with true relative file paths
        self.input_dir = os.path.split(self.input_file)[0]
        self.paths_relative = self.df['path']
        if self.view == 'overhead' and self.rule == 'cvusa':
            # Convert CVUSA streetview surface path to overhead path
            self.paths_relative = self.paths_relative.str.replace(
                'streetview/cutouts', 'streetview_aerial/' + str(self.zoom),
                n=1, regex=False)
            self.paths_relative = self.paths_relative.str.replace(
                '_90.jpg', '.jpg', n=1, regex=False)
            self.paths_relative = self.paths_relative.str.replace(
                '_270.jpg', '.jpg', n=1, regex=False)
            # Convert CVUSA flickr surface path to overhead path
            self.paths_relative = self.paths_relative.str.replace(
                'flickr', 'flickr_aerial/' + str(self.zoom), n=1, regex=False)
            self.paths_relative = self.paths_relative.str.replace(
                r'[0-9]+@N[0-9]+_[0-9]+_', '', n=1, regex=True)
            self.paths_relative = self.paths_relative.str.replace(
                '.png', '.jpg', n=1, regex=False)
        if self.view == 'overhead' and self.rule == 'witw':
            # Convert WITW surface path to overhead path
            self.paths_relative = self.paths_relative.str.replace(
                'surface', 'overhead', n=1, regex=False)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data = {}
        data['idx'] = idx
        data['path_csv'] = self.input_dir
        data['path_listed'] = self.df.iloc[idx]['path']
        data['path_relative'] = self.paths_relative[idx]
        data['path_absolute'] = os.path.join(data['path_csv'],
                                             data['path_relative'])

        if len(self.df.columns) > 3:
            data['vector'] = torch.tensor(self.df.iloc[idx, 3:].values.astype('float32'))

        data['image'] = Image.open(data['path_absolute'])
        if self.transform is not None:
            data['image'] = self.transform(data['image'])

        return data

    def populate_latlon(self):
        """
        Populate latitude and longitude columns from path column.
        This only works if entries follow format of CVUSA dataset.
        """
        def select_fields(s):
            l = s.split('_')
            if len(l) == 3:
                return '_'.join(l[:2])
            if len(l) == 4:
                return '_'.join(l[2:])
            else:
                raise Exception('! Invalid string in populate_latlon().')
        names = self.df['path'].apply(lambda x: os.path.splitext(os.path.split(x)[1])[0])
        strings = names.apply(select_fields)
        self.df[['lat', 'lon']] = strings.str.split('_', expand=True)


def get_transform(view='surface', preprocess=True, finalprocess=True):
    """
    Return image transform
    """
    transform_list = []
    if preprocess:
        if view == 'surface':
            transform_list.extend([
                torchvision.transforms.Resize((256,256)),
                torchvision.transforms.CenterCrop(224),
            ])
        elif view == 'overhead':
            transform_list.append(
                torchvision.transforms.Resize((224,224))
            )
        else:
            raise Exception('! Invalid view in get_transform().')
    if finalprocess:
        transform_list.extend([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
        ])
    transform = torchvision.transforms.Compose(transform_list)
    return transform


def load_model(view='surface', arch='alexnet'):
    """
    Based on https://github.com/CSAILVision/places365/blob/master/run_placesCNN_basic.py by Bolei Zhou
    """
    model_path = '../weights/%s_%s.pth.tar' % (arch, view)
    model = torchvision.models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    state_dict = {str.replace(k,'module.',''): v for k,v in checkpoint['state_dict'].items()}
    model.load_state_dict(state_dict)
    model.eval()
    return model


def preprocess(input_file, output_dir, view='surface', rule='cvusa'):
    """
    Preprocess images (resize and crop, if applicable),
    saving output to a new folder.
    """
    transform = get_transform(view, preprocess=True, finalprocess=False)
    dataset = OneDataset(input_file, view=view, rule=rule, transform=transform)

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        output_path = os.path.join(output_dir, data['path_relative'])
        output_subdir = os.path.split(output_path)[0]
        os.makedirs(output_subdir, exist_ok=True)
        data['image'].save(output_path)


def extract_features(input_file, output_file, view='surface', rule='cvusa',
                     batch_size=256, num_workers=12,
                     populate_latlon=False):
    """
    Use the model to generate a feature vector for each image,
    saving these in a new CSV file.
    """
    transform = get_transform(view, preprocess=False, finalprocess=True)
    dataset = OneDataset(input_file, view=view, rule=rule, transform=transform)
    if populate_latlon:
        dataset.populate_latlon()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    model = load_model(view).to(device)
    if device_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    torch.set_grad_enabled(False)

    # Generate feature vectors
    feat_vecs = None
    num_batches = len(dataset)//batch_size + (len(dataset)%batch_size > 0)
    for batch, data in tqdm.tqdm(enumerate(loader), total=num_batches):
        images = data['image'].to(device)
        feat_vecs_part = model(images)
        if feat_vecs is None:
            feat_vecs = feat_vecs_part
        else:
            feat_vecs = torch.cat((feat_vecs, feat_vecs_part), dim=0)

    # Load feature vectors into DataFrame, and save as CSV
    feat_vecs_df = pd.DataFrame(feat_vecs.cpu().numpy())
    dataset.df = pd.concat([dataset.df.iloc[:, :3], feat_vecs_df], axis=1)
    dataset.df.to_csv(output_file, header=False, index=False)


def train(input_file, view='overhead', rule='cvusa', arch='alexnet',
          batch_size=64, num_workers=12,
          val_quantity=10, num_epochs=999999):
    """
    Train a model to predict a feature vector from corresponding image.
    In particular, train a model to predict scene vector from overhead image.
    """
    transform = get_transform(view, preprocess=False, finalprocess=True)

    # Data
    dataset = OneDataset(input_file, view=view, rule=rule, transform=transform)
    train_set, val_set = torch.utils.data.random_split(dataset, [len(dataset) - val_quantity, val_quantity])
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)

    # Model, loss, optimizer (Note: Init model w/ surface weights regardless)
    model = load_model(view='surface', arch=arch).to(device)
    if device_parallel and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    loss_func = torch.nn.PairwiseDistance()
    optimizer = torch.optim.SGD(model.parameters(), lr=1E-5, momentum=0.9)
    #optimizer = torch.optim.Adam(model.parameters(), lr=1E-5)

    # Optionally resume training from where it left off
    # Note: Add "resume=False" to arguments of train()
    # To do: Load checkpoint
    # if resume:
    #     init_epoch = checkpoint['epoch']
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # else:
    #     init_epoch = 0

    # Loop through epochs
    best_loss = None
    for epoch in range(num_epochs):
        print('Epoch %d, %s' % (epoch + 1, time.ctime(time.time())))

        for phase in ['train', 'val']:
            running_count = 0
            running_loss = 0.

            if phase == 'train':
                loader = train_loader
                model.train()
            elif phase == 'val':
                loader = val_loader
                model.eval()

            # Loop through batches of data
            for batch, data in enumerate(loader):
                images = data['image'].to(device)
                target_vectors = data['vector'].to(device)

                with torch.set_grad_enabled(phase == 'train'):

                    # Forward and loss (train and val)
                    infer_vectors = model(images)
                    loss = torch.sum(loss_func(infer_vectors, target_vectors))\
                           / math.sqrt(target_vectors.size(0))

                    # Backward and optimization (train only)
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                count = target_vectors.size(0)
                running_count += count
                running_loss += loss.item()

            print('  %5s: avg loss = %f' % (phase, running_loss / running_count))

        # Save weights if this is the lowest observed validation loss
        if best_loss is None or running_loss / running_count < best_loss:
            print('-------> new best')
            best_loss = running_loss / running_count
            checkpoint_path = '../weights/%s_%s.pth.tar' % (arch, view)
            checkpoint = {
                'view': view,
                'arch': arch,
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_loss,
            }
            torch.save(checkpoint, checkpoint_path)


def metrics(surface_file, overhead_file):
    """
    Given a CSV file of surface feature vectors and a CSV file of overhead
    feature vectors for the same places in the same order, determine
    performance metrics for ranking overhead matches for each surface image.
    """
    surface_dataset = OneDataset(surface_file, view='surface', transform=None)
    surface_vectors = torch.tensor(surface_dataset.df.iloc[:, 3:].values, device=device, dtype=torch.float32)
    surface_dataset = None
    overhead_dataset = OneDataset(overhead_file, view='overhead', transform=None)
    overhead_vectors = torch.tensor(overhead_dataset.df.iloc[:, 3:].values, device=device, dtype=torch.float32)
    overhead_dataset = None

    # Measure performance
    count = surface_vectors.size(0)
    ranks = np.zeros([count], dtype=int)
    for idx in tqdm.tqdm(range(count)):
        surface_vector = torch.unsqueeze(surface_vectors[idx], 0)
        distances = torch.pow(torch.sum(torch.pow(overhead_vectors - surface_vector, 2), dim=1), 0.5)
        distance = distances[idx]
        ranks[idx] = torch.sum(torch.le(distances, distance)).item()
    top_one = np.sum(ranks <= 1) / count * 100
    top_five = np.sum(ranks <= 5) / count * 100
    top_ten = np.sum(ranks <= 10) / count * 100
    top_percent = np.sum(ranks * 100 <= count) / count * 100
    mean = np.mean(ranks)
    median = np.median(ranks)

    # Print performance
    print('Top  1: {:.2f}%'.format(top_one))
    print('Top  5: {:.2f}%'.format(top_five))
    print('Top 10: {:.2f}%'.format(top_ten))
    print('Top 1%: {:.2f}%'.format(top_percent))
    print('Avg. Rank: {:.2f}'.format(mean))
    print('Med. Rank: {:.2f}'.format(median))
    print('Locations: {}'.format(count))


def example_features(path, view='surface'):
    transform = get_transform(view)
    model = load_model(view)
    img = Image.open(path)
    batch = torch.autograd.Variable(transform(img).unsqueeze(0))
    logit = model.forward(batch)
    h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()
    print(h_x)
    print(torch.argmax(h_x))


if __name__ == '__main__':
    pass
