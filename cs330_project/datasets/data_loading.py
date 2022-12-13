# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:05:40 2021

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/MCG-NJU/VideoMAE
"""

import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms.functional as TF

from cs330_project.datasets.video_augmentation import (
    GroupMultiScaleCrop,
    GroupRandomCrop,
    GroupCenterCrop
)


class VideoDatasetType:
    TINY_VIRAT = "tiny_virat"


DATASET_TO_NUM_CLASSES = {
    VideoDatasetType.TINY_VIRAT: 10
}

IMG_DATASET_TO_IMG_SIZE = {
    VideoDatasetType.TINY_VIRAT: 32
}


class RawDataset(Dataset):
    def __init__(
            self,
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
    def __init__(
            self,
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


class MaskGenerator:
    def __init__(
            self,
            *,
            num_patches,
            mask_ratio):

        self.num_patches = num_patches
        self.num_masked_patches = int(
            mask_ratio * self.num_patches)
        self.num_visible_patches = self.num_patches - \
            self.num_masked_patches

        self.rng = np.random.default_rng()

    def __call__(self):
        shuffle_indices = np.arange(self.num_patches)
        self.rng.shuffle(shuffle_indices)
        unshuffle_indices = np.argsort(shuffle_indices)

        masked_indices = shuffle_indices[:self.num_visible_patches]

        mask = np.ones([self.num_patches])
        mask[:self.num_visible_patches] = 0
        mask = np.take(mask, unshuffle_indices, axis=0)
        
        mask = mask.astype(np.bool)
        shuffle_indices = shuffle_indices.astype(np.int64)
        unshuffle_indices = unshuffle_indices.astype(np.int64)
        masked_indices = masked_indices.astype(np.int64)

        return mask, shuffle_indices, unshuffle_indices, masked_indices


class VideoAugmentTransform:
    CROP_TYPE_MULTI_SCALE = 0
    CROP_TYPE_RANDOM = 1
    CROP_TYPE_CENTER = 2
    
    def __init__(
            self,
            input_size,
            crop_type=0):

        self.input_mean = [0.485, 0.456, 0.406]  # ImageNet default mean
        self.input_std = [0.229, 0.224, 0.225]  # ImageNet default std
        crop_transform = {
            self.CROP_TYPE_MULTI_SCALE: GroupMultiScaleCrop,
            self.CROP_TYPE_RANDOM: GroupRandomCrop,
            self.CROP_TYPE_CENTER: GroupCenterCrop
        }[crop_type]
        self.resize = crop_transform(input_size)

    def __call__(self, images_per_frame):
        images_per_frame = self.resize(images_per_frame)  # PIL Image, t
        images_per_frame = np.stack(images_per_frame, axis=0)  # t, h, w, c
        images_per_frame = [TF.to_tensor(image) for image in images_per_frame] # t, c, h, w
        for image in images_per_frame:
            TF.normalize(image, self.input_mean, self.input_std, inplace=True)
        images_per_frame = torch.stack(images_per_frame, dim=0)
        images_per_frame = images_per_frame.permute((1, 0, 2, 3))  # c, t, h, w

        return images_per_frame


class MaskedVideoAutoencoderTransform:
    def __init__(
        self,
        *,
        input_size,
        num_patches,
        mask_ratio=0.75,
        crop_type=0):
        
        self.augment = VideoAugmentTransform(
            input_size=input_size,
            crop_type=crop_type)
        self.mask_generator = MaskGenerator(
            num_patches=num_patches,
            mask_ratio=mask_ratio)

    def __call__(self, images_per_frame):
        images_per_frame = self.augment(images_per_frame)
        mask_info = self.mask_generator()
        
        return images_per_frame, mask_info


def get_dataloaders(
        datasets,
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
