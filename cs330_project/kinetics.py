# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 00:07:14 2022

@author: Shahir, Faraz, Pratyush
Modified from: https://github.com/MCG-NJU/VideoMAE
"""

import os
import json

import numpy as np

from torch.utils.data import Dataset

import decord
from PIL import Image


class VideoMAE(Dataset):
    def __init__(
            self,
            root,
            setting,
            train=True,
            test_mode=False,
            name_pattern='img_%05d.jpg',
            video_ext='mp4',
            is_color=True,
            modality='rgb',
            num_segments=1,
            num_crop=1,
            new_length=1,
            new_step=1,
            label_map=None,
            transform=None,
            temporal_jitter=False,
            video_loader=False,
            use_decord=False,
            lazy_init=False):

        super().__init__()

        self.root = root
        self.setting = setting
        self.train = train
        self.test_mode = test_mode
        self.is_color = is_color
        self.modality = modality
        self.num_segments = num_segments
        self.num_crop = num_crop
        self.new_length = new_length
        self.new_step = new_step
        self.skip_length = self.new_length * self.new_step
        self.temporal_jitter = temporal_jitter
        self.name_pattern = name_pattern
        self.video_loader = video_loader
        self.video_ext = video_ext
        self.use_decord = use_decord
        self.transform = transform
        self.lazy_init = lazy_init
        self.labelMap = label_map

        if not self.lazy_init:
            self.clips, self.codingToLabel = self._make_dataset(
                root, setting, self.labelMap)
            if len(self.clips) == 0:l
                raise (RuntimeError("Found 0 video clips in subfolders of: " + 127 rond  
                 \ + "\n"
                                                                                      "Check your data directory (opt.data-dir)."))

    def __getitem__(self, index):
        directory, target = self.clips[index]
        if self.video_loader:
            if '.' in directory.split('/')[-1]:
                # data in the "setting" file already have extension, e.g., demo.mp4
                video_name = directory
            else:
                # data in the "setting" file do not have extension, e.g., demo
                # So we need to provide extension (i.e., .mp4) to complete the file name.
                video_name = '{}.{}'.format(directory, self.video_ext)

            decord_vr = decord.VideoReader(video_name, num_threads=1)
            duration = len(decord_vr)

        segment_indices, skip_offsets = self._sample_train_indices(duration)

        images = self._video_tsn_decord_batch_loader(
            directory, decord_vr, duration, segment_indices, skip_offsets)

        process_data, mask = self.transform((images, None))  # T*C,H,W
        process_data = process_data.view((self.new_length, 3) +
                                         process_data.size()[-2:]).transpose(0, 1)  # T*C,H,W -> T,C,H,W -> C,T,H,W

        return (process_data, mask)

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self, directory, setting, labelMap):
        # Opening JSON file
        f = open(setting)
        data = json.load(f)
        tubes = data["tubes"]

        labelsToCodingSet = set()
        labelsToCoding = {}
        codingToLabels = {}

        # we need to reuse labelMap if already created in training setup
        if labelMap == None:
            for tube in tubes:
                label = tube["label"]
                if label[0] not in labelsToCodingSet:
                    labelsToCodingSet.add(label[0])
            counter = 0
            for label in labelsToCodingSet:
                labelsToCoding[label] = counter
                codingToLabels[counter] = label
                counter = counter + 1
        else:
            codingToLabels = labelMap
            for code in codingToLabels:
                label = codingToLabels[code]
                labelsToCoding[label] = code

        if not os.path.exists(setting):
            raise (RuntimeError(
                "Setting file %s doesn't exist. Check opt.train-list and opt.val-list. " % (setting)))

        clips = []
        for tube in tubes:
            clip_path = directory + tube["path"]
            label = tube["label"][0]
            target = labelsToCoding[label]
            item = (clip_path, target)
            clips.append(item)
        return clips, codingToLabels

    def _sample_train_indices(self, num_frames):
        average_duration = (
            num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(list(range(self.num_segments)),
                                  average_duration)
            offsets = offsets + np.random.randint(average_duration,
                                                  size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(np.random.randint(
                num_frames - self.skip_length + 1,
                size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = np.random.randint(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(
                self.skip_length // self.new_step, dtype=int)
        return offsets + 1, skip_offsets

    def _video_tsn_decord_batch_loader(self, directory, video_reader, duration, indices, skip_offsets):
        sampled_list = []
        frame_id_list = []
        for seg_ind in indices:
            offset = int(seg_ind)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_id_list.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step
        try:
            video_data = video_reader.get_batch(frame_id_list).asnumpy()
            sampled_list = [Image.fromarray(video_data[vid, :, :, :]).convert('RGB') for vid, _ in
                            enumerate(frame_id_list)]
        except:
            raise RuntimeError(
                'Error occured in reading frames {} from video {} of duration {}.'.format(frame_id_list, directory,
                                                                                          duration))
        return sampled_list
