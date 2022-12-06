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


class MaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.num_frames, self.height, self.width = input_size
        self.num_patches_per_frame = self.height * self.width
        self.num_masked_patches_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.num_visible_patches_per_frame = self.num_patches_per_frame - self.num_masked_patches_per_frame
        self.total_masks = self.num_frames * self.num_masked_patches_per_frame
        self.rng = np.random.default_rng()

    def __call__(self):
        shuffle_indices = np.arange(self.num_patches_per_frame)
        self.rng.shuffle(mask_per_frame)
        unshuffle_indices = np.argsort(shuffle_indices)
        
        masked_indices = shuffle_indices[:, :self.num_visible_patches_per_frame]
        
        mask_per_frame = np.ones([self.num_patches_per_frame])
        mask_per_frame[:, :self.num_visible_patches_per_frame] = 0
        mask_per_frame = np.take(mask_per_frame, unshuffle_indices, axis=0) # [H * W]
        mask = np.tile(mask_per_frame, (self.num_frames, 1)).flatten() # [H * W * T, 1]
        
        return mask, shuffle_indices, unshuffle_indices, masked_indices


class MaskedVideoAutoencoderTransform:
    def __init__(self, input_size, mask_window_size=1, tube_masking=False, mask_ratio=0.75):
        self.input_mean = [0.485, 0.456, 0.406]  # ImageNet default mean
        self.input_std = [0.229, 0.224, 0.225]  # ImageNet default std
        normalize = GroupNormalize(self.input_mean, self.input_std)
        self.train_augmentation = GroupMultiScaleCrop(
            input_size, [1, .875, .75, .66])
        self.augment = transforms.Compose([
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        if tube_masking:
            self.masked_position_generator = MaskingGenerator(
                mask_window_size, mask_ratio
            )

    def __call__(self, images):
        process_data, _ = self.augment(images)
        return process_data, *self.masked_position_generator()


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
