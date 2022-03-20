#!/usr/bin/env python

import os
import re
import torch
import torchvision
import pandas as pd
from PIL import Image


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
        self.df = pd.read_csv(
            self.input_file, header=None,
            names=['path', 'lat', 'lon']
        )
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        data = {}
        data['path_csv'] = os.path.split(self.input_file)[0]
        data['path_listed'] = self.df.iloc[idx]['path']
        data['path_relative'] = data['path_listed']
        if self.view == 'overhead' and self.rule == 'cvusa':
            # Convert CVUSA streetview surface path to overhead path
            data['path_relative'] = data['path_relative'].replace(
                'streetview/cutouts', 'streetview_aerial/' + str(self.zoom), 1)
            data['path_relative'] = data['path_relative'].replace(
                '_90.jpg', '.jpg', 1)
            data['path_relative'] = data['path_relative'].replace(
                '_270.jpg', '.jpg', 1)
            # Convert CVUSA flickr surface path to overhead path
            data['path_relative'] = data['path_relative'].replace(
                'flickr', 'flickr_aerial/' + str(self.zoom), 1)
            data['path_relative'] = re.sub(
                r'[0-9]+@N[0-9]+_[0-9]+_', '', data['path_relative'])

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
                raise Exception('! Invalid string in ' + __name__)
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
            raise Exception('! Invalid view in ' + __name__)
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

    transform = get_transform(view, preprocess=True, finalprocess=False)
    dataset = OneDataset(input_file, view=view, transform=transform)
    dataset.populate_latlon()

    for idx in range(len(dataset)):
        print(dataset[idx])


def save_features(view='surface'):

    transform = get_transform(view)

    model = load_model(view)

    img = Image.open('../example/60949863@N02_7984662477_43.533763_-89.290620.jpg')
    batch = torch.autograd.Variable(transform(img).unsqueeze(0))
    logit = model.forward(batch)
    h_x = torch.nn.functional.softmax(logit, 1).data.squeeze()
    print(h_x)
    print(torch.argmax(h_x))


if __name__ == '__main__':
    preprocess('/local_data/crossviewusa/streetview_images.txt', '../temp', view='overhead')
    #preprocess('/local_data/crossviewusa/sample/streetview_images.txt', '../temp', view='overhead')
    #save_features()
