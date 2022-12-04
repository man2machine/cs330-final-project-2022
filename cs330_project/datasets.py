# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:40 2021

@author: Shahir
"""

import os
import math
from functools import partial

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

import torchvision
from torchvision import transforms
import torchvision.transforms.functional as TF

class VideoDatasetType:
    TEMP = "temp"

DATASET_TO_NUM_CLASSES = {
    VideoDatasetType.TEMP: 10
}

IMG_DATASET_TO_IMG_SIZE = {
    VideoDatasetType.TEMP: 32
}

class RawDataset(Dataset):
    def __init__(self,
                 x_data,
                 y_data=None,
                 metadata=None):
        self.labeled = y_data is not None
        self.x_data = x_data
        self.y_data = y_data
        self.metadata = metadata

    def __getitem__(self, index):
        if self.labeled:
            return (self.x_data[index], self.y_data[index])
        
        return self.x_data[index]

    def __len__(self):
        return len(self.x_data)

class TransformDataset(Dataset):
    def __init__(self,
                 dataset,
                 labeled=True,
                 transform_func=None):
        self.dataset = dataset
        self.transform_func = transform_func
        self.labeled = labeled

    def __getitem__(self, index):
        if self.labeled:
            inputs, label = self.dataset[index]
            if self.transform_func:
                inputs = self.transform_func(inputs)
            return (inputs, label)
        else:
            inputs = self.dataset[index]
            inputs = self.transform_func(inputs)
            return inputs
    
    def __len__(self):
        return len(self.dataset)

def apply_augmentation(datasets,
                         dataset_type,
                         augment=False):
    if dataset_type == ImageDatasetType.MNIST:
        train_transform = transforms.Compose([
            transforms.RandomApply([
                transforms.RandomAffine(degrees=10,
                                        translate=(0.1, 0.1),
                                        scale=(0.9, 1.1),
                                        shear=10)], p=0.9),
            transforms.ToTensor()])
    
        test_transform = transforms.Compose([
            transforms.ToTensor()])
    
    else:
        raise ValueError()
    
    if not augment:
        train_transform = test_transform

    train_dataset = TransformDataset(
        datasets['train'],
        transform_func=train_transform)
    test_dataset = TransformDataset(
        datasets['test'],
        transform_func=test_transform)
    
    return {'train': train_dataset,
            'test': test_dataset}

def get_dataloaders(datasets,
                    train_batch_size,
                    test_batch_size,
                    num_workers=4,
                    pin_memory=False):
    train_dataset = datasets['train']
    train_loader = DataLoader(train_dataset,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    test_dataset = datasets['test']
    test_loader = DataLoader(test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory)
    test_shuffle_loader = DataLoader(test_dataset,
        batch_size=test_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory)
    
    return {'train': train_loader,
            'test': test_loader,
            'test_shuffle': test_shuffle_loader}
