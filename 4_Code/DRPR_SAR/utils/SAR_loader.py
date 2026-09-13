from torchvision.models import resnet18, vgg13_bn, densenet121, wide_resnet50_2, alexnet, efficientnet_b0, \
    shufflenet_v2_x2_0, mobilenet_v3_large, resnet50, resnet101
import sys
import argparse
from typing import Any
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
# from classifiers.models import get_model
# import torchattacks
# from thop import profile, clever_format
sys.path.append('.')
from utils.randaugment4fixmatch import  RandAugmentMC

def load_data(args, adv_batch_size):
    resize_shape = 128#128
    _, test_loader = get_dataloader(args.domain, adv_batch_size, resize_shape)
    # x_val, y_val = next(iter(test_loader))
    # print(f'x_val shape: {x_val.shape}')
    # x_val, y_val = x_val.contiguous().requires_grad_(True), y_val.contiguous()
    # print(f'x (min, max): ({x_val.min()}, {x_val.max()})')

    return test_loader


def get_dataloader(dataset, bs, size, train_path, test_path,  mean, std, shuffle=True):
    dataset = dataset
    train_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        RandAugmentMC(n=2, m=10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    test_transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


    train_dataset = ImageFolder(train_path, transform=train_transform)
    test_datset = ImageFolder(test_path, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=shuffle, num_workers=0)
    test_loader = DataLoader(test_datset, batch_size=bs, shuffle=shuffle, num_workers=0)

    return train_loader, test_loader

def get_model(name, num_class):
    name = '_' + name
    return globals()[name](num_class=num_class)