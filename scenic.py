#!/usr/bin/env python

import os
import tqdm
import torch
import torchvision
import pandas as pd
from PIL import Image

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
        self.df = pd.read_csv(
            self.input_file, header=None,
            names=['path', 'lat', 'lon']
        )

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


def preprocess(input_file, output_dir, view='surface'):
    """
    Preprocess images (resize and crop, if applicable),
    saving output to a new folder.
    """
    transform = get_transform(view, preprocess=True, finalprocess=False)
    dataset = OneDataset(input_file, view=view, transform=transform)
    dataset.populate_latlon()

    for idx in tqdm.tqdm(range(len(dataset))):
        data = dataset[idx]
        output_path = os.path.join(output_dir, data['path_relative'])
        output_subdir = os.path.split(output_path)[0]
        os.makedirs(output_subdir, exist_ok=True)
        data['image'].save(output_path)
    # To do: save CSV file with latlon


def extract_features_from_images(input_file, output_file, view='surface',
                                 batch_size=64, num_workers=8):
    """
    Use the model to generate a feature vector for each image,
    saving these in a new CSV file.
    """
    transform = get_transform(view, preprocess=False, finalprocess=True)
    dataset = OneDataset(input_file, view=view, transform=transform)
    dataset.populate_latlon()
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers)
    model = load_model(view).to(device)
    torch.set_grad_enabled(False)

    feat_vecs = None
    for batch, data in enumerate(loader):
        data = data.to(device)
        feat_vecs_part = model(data)
        if feat_vecs is None:
            feat_vecs = feat_vecs_part
        else:
            feat_vecs = torch.cat((feat_vecs, feat_vecs_part), dim=0)

    # To do: save CSV file after adding features (and latlon?)


def example_features(path='../example/60949863@N02_7984662477_43.533763_-89.290620.jpg', view='surface'):
    transform = get_transform(view)
    model = load_model(view)
    img = Image.open(path)
    batch = torch.autograd.Variable(transform(img).unsqueeze(0))
    logit = model.forward(batch)
    h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()
    print(h_x)
    print(torch.argmax(h_x))


if __name__ == '__main__':
    choice = 2
    if choice == 0:
        example_features()
    elif choice == 1:
        preprocess('/local_data/crossviewusa/streetview_images.txt',
                   '/local_data/crossviewusa/preprocessed', view='surface')
    elif choice == 2:
        preprocess('/local_data/crossviewusa/flickr_images.txt',
                   '/local_data/crossviewusa/preprocessed', view='overhead')
    elif choice == 3:
        pass
