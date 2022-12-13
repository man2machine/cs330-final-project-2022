# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 22:08:39 2022

@author: Shahir, Faraz, Pratyush
"""

import random
import numbers

import numpy as np

from torchvision import transforms
import torchvision.transforms.functional as TF

from PIL import Image

from cs330_project.utils import make_pair_shape

class GroupRandomCrop:
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img_group):
        w, h = img_group[0].size
        th, tw = self.size

        out_imgs = list()

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)

        for img in img_group:
            assert (img.size[0] == w and img.size[1] == h)
            if w == tw and h == th:
                out_imgs.append(img)
            else:
                out_imgs.append(img.crop((x1, y1, x1 + tw, y1 + th)))

        return out_imgs


class GroupCenterCrop(object):
    def __init__(self, size):
        self.worker = transforms.CenterCrop(size)

    def __call__(self, img_group):
        return [self.worker(img) for img in img_group]


class GroupNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        rep_mean = self.mean * (tensor.size()[0] // len(self.mean))
        rep_std = self.std * (tensor.size()[0] // len(self.std))

        # TODO: make efficient
        for t, m, s in zip(tensor, rep_mean, rep_std):
            t.sub_(m).div_(s)

        return tensor


class GroupMultiScaleCrop:
    def __init__(
            self,
            input_size,
            scales=None,
            max_distort=1,
            fix_crop=True,
            more_fix_crop=True):

        self.scales = scales if scales is not None else [1, 0.875, 0.75, 0.66]
        self.max_distort = max_distort
        self.fix_crop = fix_crop
        self.more_fix_crop = more_fix_crop
        self.input_size = make_pair_shape(input_size)
        self.interpolation = Image.BILINEAR

    def __call__(self, img_group):
        im_size = img_group[0].size

        crop_w, crop_h, offset_w, offset_h = self._sample_crop_size(im_size)
        crop_img_group = [img.crop(
            (offset_w, offset_h, offset_w + crop_w, offset_h + crop_h)) for img in img_group]
        ret_img_group = [img.resize((self.input_size[0], self.input_size[1]), self.interpolation) for img in
                         crop_img_group]

        return ret_img_group

    def _sample_crop_size(self, img_size):
        img_w, img_h = img_size[0], img_size[1]

        # find a crop size
        base_size = min(img_w, img_h)
        crop_sizes = [int(base_size * x) for x in self.scales]
        crop_h = [self.input_size[1] if abs(
            x - self.input_size[1]) < 3 else x for x in crop_sizes]
        crop_w = [self.input_size[0] if abs(
            x - self.input_size[0]) < 3 else x for x in crop_sizes]

        pairs = []
        for i, h in enumerate(crop_h):
            for j, w in enumerate(crop_w):
                if abs(i - j) <= self.max_distort:
                    pairs.append((w, h))

        crop_pair = random.choice(pairs)
        if not self.fix_crop:
            w_offset = random.randint(0, img_w - crop_pair[0])
            h_offset = random.randint(0, img_h - crop_pair[1])
        else:
            w_offset, h_offset = self._sample_fix_offset(
                img_w, img_h, crop_pair[0], crop_pair[1])

        return crop_pair[0], crop_pair[1], w_offset, h_offset

    def _sample_fix_offset(
            self,
            img_w,
            img_h,
            crop_w,
            crop_h):

        offsets = self.fill_fix_offset(
            self.more_fix_crop, img_w, img_h, crop_w, crop_h)
        return random.choice(offsets)

    @staticmethod
    def fill_fix_offset(
            more_fix_crop,
            img_w,
            img_h,
            crop_w,
            crop_h):

        w_step = (img_w - crop_w) // 4
        h_step = (img_h - crop_h) // 4

        ret = list()
        ret.append((0, 0))  # upper left
        ret.append((4 * w_step, 0))  # upper right
        ret.append((0, 4 * h_step))  # lower left
        ret.append((4 * w_step, 4 * h_step))  # lower right
        ret.append((2 * w_step, 2 * h_step))  # center

        if more_fix_crop:
            ret.append((0, 2 * h_step))  # center left
            ret.append((4 * w_step, 2 * h_step))  # center right
            ret.append((2 * w_step, 4 * h_step))  # lower center
            ret.append((2 * w_step, 0 * h_step))  # upper center

            ret.append((1 * w_step, 1 * h_step))  # upper left quarter
            ret.append((3 * w_step, 1 * h_step))  # upper right quarter
            ret.append((1 * w_step, 3 * h_step))  # lower left quarter
            ret.append((3 * w_step, 3 * h_step))  # lower righ quarter
        return ret


class Stack:
    def __init__(self, roll=False):
        self.roll = roll

    def __call__(self, img_group):
        if img_group[0].mode == 'L': # 8-bit black and white
            return np.concatenate([np.expand_dims(x, 2) for x in img_group], axis=2)
        elif img_group[0].mode == 'RGB': # 3 x 8-bit color
            return np.concatenate(img_group, axis=2)
