# based onhttps://github.com/aanna0701/SPT_LSA_ViT
import numpy as np
import os
import random
import torch
from torch.utils.data import IterableDataset
import time
import imageio


def get_images(paths, labels, nb_samples=None, shuffle=True):
    """
    Takes a set of character folders and labels and returns paths to image files
    paired with labels.
    Args:
        paths: A list of character folders
        labels: List or numpy array of same length as paths
        nb_samples: Number of images to retrieve per character
    Returns:
        List of (label, image_path) tuples
    """
    import binascii
    if nb_samples is not None:
        sampler = lambda x: random.sample(x, nb_samples)
    else:
        sampler = lambda x: x

    images_labels = []
    for i, path in zip(labels, paths):
        image = sampler(os.listdir(path))
        images_labels.append((i, os.path.join(path, str(image[0].decode('ASCII')))))

    if shuffle:
        random.shuffle(images_labels)

    return np.array(images_labels)


class DataGenerator(IterableDataset):
    """
    Data Generator capable of generating batches of Omniglot data.
    A "class" is considered a class of omniglot digits.
    """

    def __init__(
            self,
            num_classes,
            num_samples_per_class,
            batch_type,
            config={},
            device=torch.device("cpu"),
            cache=True,
    ):
        """
        Args:
            num_classes: Number of classes for classification (N-way)
            num_samples_per_class: num samples to generate per class in one batch (K+1)
            batch_size: size of meta batch size (e.g. number of functions)
            batch_type: train/val/test
        """
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes

        data_folder = config.get("data_folder", "./omniglot_resized")
        self.img_size = config.get("img_size", (28, 28))

        self.dim_input = np.prod(self.img_size)
        self.dim_output = self.num_classes

        character_folders = [
            os.path.join(data_folder, family, character)
            for family in os.listdir(data_folder)
            if os.path.isdir(os.path.join(data_folder, family))
            for character in os.listdir(os.path.join(data_folder, family))
            if os.path.isdir(os.path.join(data_folder, family, character))
        ]

        random.seed(1)
        random.shuffle(character_folders)
        num_val = 100
        num_train = 1100
        self.metatrain_character_folders = character_folders[:num_train]
        self.metaval_character_folders = character_folders[num_train: num_train + num_val]
        self.metatest_character_folders = character_folders[num_train + num_val:]
        self.device = device
        self.image_caching = cache
        self.stored_images = {}

        if batch_type == "train":
            self.folders = self.metatrain_character_folders
        elif batch_type == "val":
            self.folders = self.metaval_character_folders
        else:
            self.folders = self.metatest_character_folders

    def image_file_to_array(self, filename, dim_input):
        """
        Takes an image path and returns numpy array
        Args:
            filename: Image filename
            dim_input: Flattened shape of image
        Returns:
            1 channel image
        """
        if self.image_caching and (filename in self.stored_images):
            return self.stored_images[filename]
        image = imageio.imread(filename)
        image = image.reshape([dim_input])
        image = image.astype(np.float32) / 255.0
        image = 1.0 - image
        if self.image_caching:
            self.stored_images[filename] = image
        return image

    def shuffleQuerySet(self, all_image_batches, all_label_batches, N):

        numList = list(range(0, N))
        np.random.shuffle(numList)

        query_images = [None] * N
        query_labels = [None] * N

        for i in range(0, N):
            query_images[i] = all_image_batches[-1][numList[i]]
            query_labels[i] = all_label_batches[-1][numList[i]]

        all_image_batches[-1] = query_images
        all_label_batches[-1] = query_labels

    def _sample(self):
        """
        Samples a batch for training, validation, or testing
        Args:
            does not take any arguments
        Returns:
            A tuple of (1) Image batch and (2) Label batch:
                1. image batch has shape [K+1, N, 784] and
                2. label batch has shape [K+1, N, N]
            where K is the number of "shots", N is number of classes
        Note:
            1. The numpy functions np.random.shuffle and np.eye (for creating)
            one-hot vectors would be useful.

            2. For shuffling, remember to make sure images and labels are shuffled
            in the same order, otherwise the one-to-one mapping between images
            and labels may get messed up. Hint: there is a clever way to use
            np.random.shuffle here.

            3. The value for `self.num_samples_per_class` will be set to K+1
            since for K-shot classification you need to sample K supports and
            1 query.
        """

        #############################
        #### YOUR CODE GOES HERE ####
        # Extract second folder from folder strings
        samples = np.ndarray(shape=(4,))
        # Placeholder
        # string = "a" * 20
        o = len('./omniglot_resized/')
        families = {}

        for string in self.folders:
            class_name = string[o:o + (string[o:].find('/'))]
            # charName   = string[1 + o + (string[o:].find('/')):]
            char_folder = string
            if not class_name in families.keys():
                families[class_name] = np.asarray(char_folder)
            else:
                families[class_name] = np.hstack([families[class_name], char_folder])

        N = self.num_classes
        K = self.num_samples_per_class
        img_size = 784
        all_image_batches = np.ndarray([K, N, img_size])
        all_label_batches = np.ndarray([K, N, N])
        classes = [f for f in families.keys()]
        #  Sample N different classes
        n_classes = np.random.choice(classes, N, replace=False)
        #  Sample and load K images
        for j, cl in enumerate(n_classes):
            k_char_folders = np.random.choice(families[cl], K, replace=True)
            k_labels = [s[1 + o + (s[o:].find('/')):] for s in k_char_folders]
            k_char_paths = [char_folder for char_folder in k_char_folders]

            k_tuples = get_images(k_char_paths, k_labels, nb_samples=K)
            k_images = np.asarray([self.image_file_to_array(im[1], img_size) for im in k_tuples])
            all_image_batches[:, j, :] = k_images.copy()
            all_label_batches[:, j, :] = np.array([0] * N)
            # we should reshuffle last pair so model dont overfit but make sure that labels and images mantain same order
            all_label_batches[:, j, j] = 1

        # This shuffle query set. For some reasons it dont work well for 2 classes but is fine for more classes
        self.shuffleQuerySet(all_image_batches, all_label_batches, N)

        #############################
        return all_image_batches, all_label_batches
        #############################

    def __iter__(self):
        while True:
            yield self._sample()
