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


class TinyVIRAT(Dataset):
    def __init__(
            self,
            root_dir,
            train=True,
            num_segments=1,
            new_length=1,
            new_step=1,
            temporal_jitter=False,
            verbose=False):

        super().__init__()

        self.root_dir = root_dir
        self.train = train
        self.num_segments = num_segments
        self.new_length = new_length
        self.new_step = new_step
        self.temporal_jitter = temporal_jitter
        self.verbose = verbose
        
        self.skip_length = self.new_length * self.new_step
        
        self.settings_fname = os.path.join(self.root_dir, "tiny_train.json" if train else "tiny_test.json")
        self.data_dir = os.path.join(self.root_dir, "videos", "train" if self.train else "test")
        
        self.clips, self.label_index_to_class_name = self._make_dataset()
        self.num_channels = 3

        self.rng = np.random.default_rng()

    def __getitem__(self, index):
        video_fname, label = self.clips[index]

        video_reader = decord.VideoReader(video_fname, num_threads=1)
        duration = len(video_reader)

        segment_indices, skip_offsets = self._sample_train_indices(duration)
        
        # t, PIL image (h, w, c)
        if self.verbose:
            print("Decoding", video_fname, duration, segment_indices, skip_offsets)
        try:
            images_per_frame = self._video_tsn_decord_batch_loader(
                video_reader, duration, segment_indices, skip_offsets)
        except:
            if self.verbose:
                print("Failed to decode", video_fname, duration, segment_indices, skip_offsets)
            raise
        
        return images_per_frame, label

    def __len__(self):
        return len(self.clips)

    def _make_dataset(self):
        with open(os.path.join(self.root_dir, "tiny_train.json")) as f:
            self.train_settings = json.load(f)
        with open(os.path.join(self.root_dir, "tiny_test.json")) as f:
            self.test_settings = json.load(f)
        
        all_tubes = self.train_settings["tubes"] + self.test_settings["tubes"]
        selected_set_tubes = self.train_settings["tubes"] if self.train else self.test_settings["tubes"]

        class_names = set()
        class_name_to_label_index = {}
        label_index_to_class_name = {}
        
        for tube in all_tubes:
            label = tube["label"]
            if label[0] not in class_names:
                class_names.add(label[0])
        class_names = sorted(class_names)
        
        counter = 0
        for label in class_names:
            class_name_to_label_index[label] = counter
            label_index_to_class_name[counter] = label
            counter = counter + 1
        clips = []
        for tube in selected_set_tubes:
            clip_path = os.path.join(self.data_dir, tube["path"])
            label = tube["label"][0]
            label = class_name_to_label_index[label]
            item = (clip_path, label)
            clips.append(item)

        return clips, label_index_to_class_name

    def _sample_train_indices(self, num_frames):
        average_duration = (
            num_frames - self.skip_length + 1) // self.num_segments
        if average_duration > 0:
            offsets = np.multiply(
                list(range(self.num_segments)), average_duration)
            offsets = offsets + \
                self.rng.integers(average_duration, size=self.num_segments)
        elif num_frames > max(self.num_segments, self.skip_length):
            offsets = np.sort(self.rng.integers(
                num_frames - self.skip_length + 1, size=self.num_segments))
        else:
            offsets = np.zeros((self.num_segments,))

        if self.temporal_jitter:
            skip_offsets = self.rng.integers(
                self.new_step, size=self.skip_length // self.new_step)
        else:
            skip_offsets = np.zeros(self.skip_length // self.new_step)

        return (offsets + 1).astype(int), skip_offsets.astype(int)

    def _video_tsn_decord_batch_loader(
            self,
            video_reader,
            duration,
            segment_indices,
            skip_offsets):
        
        frame_ids = []

        for segment_index in segment_indices:
            offset = int(segment_index)
            for i, _ in enumerate(range(0, self.skip_length, self.new_step)):
                if offset + skip_offsets[i] <= duration:
                    frame_id = offset + skip_offsets[i] - 1
                else:
                    frame_id = offset - 1
                frame_ids.append(frame_id)
                if offset + self.new_step < duration:
                    offset += self.new_step

        video_data = video_reader.get_batch(frame_ids).asnumpy()
        sampled_images = [Image.fromarray(video_data[i]).convert('RGB') for i in range(len(frame_ids))]

        return sampled_images
