# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.
#
from torch.utils.data.sampler import BatchSampler
import torch
import numpy as np
from torch.utils.data.dataloader import default_collate
from collections.abc import Mapping, Sequence


class RepeatAugmentationSampler(torch.utils.data.Sampler):
    """
    Batch Sampler with Repeated Augmentations (RA)
    - dataset_len: original length of the dataset
    - batch_size
    - repetitions: instances per image
    - len_factor: multiplicative factor for epoch size
    """

    def __init__(self, dataset_len, batch_size, repetitions=1, len_factor=3.0, shuffle=False, drop_last=False):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.repetitions = repetitions
        self.len_images = int(dataset_len * len_factor)
        self.shuffle = shuffle
        self.drop_last = drop_last

    def shuffler(self):
        if self.shuffle:
            new_perm = lambda: iter(np.random.permutation(self.dataset_len))
        else:
            new_perm = lambda: iter(np.arange(self.dataset_len))
        shuffle = new_perm()
        while True:
            try:
                index = next(shuffle)
            except StopIteration:
                shuffle = new_perm()
                index = next(shuffle)
            for _ in range(self.repetitions):
                yield index

    def __iter__(self):
        shuffle = iter(self.shuffler())
        batch = []
        for _ in range(self.len_images):
            index = next(shuffle)
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    def __len__(self):
        if self.drop_last:
            return self.len_images // self.batch_size
        else:
            return (self.len_images + self.batch_size - 1) // self.batch_size

