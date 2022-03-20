#!/usr/bin/env python

import torch
import torchvision
import os
from PIL import Image


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
    save_features()
