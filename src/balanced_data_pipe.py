import glob
import os
from math import ceil, floor
from typing import Any

import PIL.Image
import numpy as np

import tensorflow as tf
from PIL import Image


class BalancedDataPipe:
    def __init__(
            self,
            data_directory: str,
            rng_seed: int = 42,
            val_split: float = 0.20,
            batch_size: int = 32,
            preprocessing_function: Any = None
    ):
        self.rng_seed = rng_seed
        self.data_dir = data_directory
        self.val_split = val_split
        self.batch_size = batch_size
        self.preprocessing_function = preprocessing_function

        self.image_files = glob.glob("{}/*/*".format(self.data_dir))
        self.total_n_samples = len(self.image_files)

        test_image = Image.open(self.image_files[0])
        img_height, img_width, img_mode = test_image.height, test_image.width, test_image.mode
        assert all(
            [
                Image.open(image).width == img_width and
                Image.open(image).height == img_height and
                Image.open(image).mode == img_mode for
                image in self.image_files
            ]
        )
        self.input_shape = (img_height, img_width, 3)

        self.labels = os.listdir(self.data_dir)
        self.n_classes = len(self.labels)

        self.n_samples_per_class = []
        for label in self.labels:
            n = len(os.listdir(os.path.join(self.data_dir, label)))
            self.n_samples_per_class.append(n)

        self.train_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.val_split,
            subset="training",
            label_mode="int",
            seed=self.rng_seed,
            image_size=(96, 96),
            batch_size=None)  # Retrieve unbatched data

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.val_split,
            subset="validation",
            label_mode="int",
            seed=self.rng_seed,
            image_size=(96, 96),
            batch_size=None)  # Retrieve unbatched data

        self.training_size: int = -1

    def _make_class_datasets(self, subsample: bool = False, seed: int = None):
        """
        Divide the original training dataset (extracted with keras.utils.image_dataset_from_directory)
        into class-specific training datasets that the full training dataset will uniformly sample from.
        """
        train_datasets = []
        max_n_samples = max(self.n_samples_per_class)

        for class_number in range(self.n_classes):
            n_samples = self.n_samples_per_class[class_number]
            class_dataset = (
                self.train_ds
                .filter(lambda features, label: label == class_number)
                .shuffle(ceil(n_samples * (1 - self.val_split)),
                         reshuffle_each_iteration=(True if seed is None else False),
                         seed=seed)
            )
            if subsample is False:
                repeat_factor = ceil(max_n_samples / n_samples)
                class_dataset = class_dataset.repeat(repeat_factor)

            train_datasets.append(class_dataset)

        return train_datasets

    def get_infinite_over_sampled_dataset(self):
        """
        Returns the "infinite" oversampling-balanced dataset.
        This dataset does not always resample the same images from the smaller classes.
        Which means that, over many epochs, there will be no images from the smaller classes that
        are seen more than the others.
        However, by design, this dataset is infinite so the steps_per_epoch parameter needs to be set manually
        as the dataset will not 'run out' at each epoch.
        """
        class_datasets = self._make_class_datasets(subsample=False)
        max_n_samples = max(self.n_samples_per_class)
        self.training_size = self.n_classes * floor(max_n_samples * (1 - self.val_split))

        over_sampled_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets,
            stop_on_empty_dataset=True,
            weights=None,  # None = equally distributed
            seed=None  # Different sampling at every epoch (problem: inconsistent dataset size from epoch to epoch)
        )

        if self.preprocessing_function is not None:
            over_sampled_ds = over_sampled_ds.map(self.preprocessing_function)

        return (over_sampled_ds.
                repeat().
                batch(self.batch_size).
                prefetch(tf.data.AUTOTUNE))

    def get_infinite_sub_sampled_dataset(self):
        """
        Returns the "infinite" subsampling-balanced dataset.
        This dataset does not always sample the same images from the bigger classes.
        Which means that, over many epochs, there will be no images from the bigger classes that
        are seen more than the others.
        However, by design, this dataset is infinite so the steps_per_epoch parameter needs to be set manually
        as the dataset will not 'run out' at each epoch.
        """
        class_datasets = self._make_class_datasets(subsample=True)
        min_n_samples = min(self.n_samples_per_class)
        self.training_size = self.n_classes * floor(min_n_samples * (1 - self.val_split))

        sub_sampled_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets,
            stop_on_empty_dataset=True,
            weights=None,  # None = equally distributed
            seed=None  # Different sampling at every epoch (problem: inconsistent dataset size from epoch to epoch)
        )

        if self.preprocessing_function is not None:
            sub_sampled_ds = sub_sampled_ds.map(self.preprocessing_function)

        return (sub_sampled_ds.
                repeat().
                batch(self.batch_size).
                prefetch(tf.data.AUTOTUNE))

    def get_fixed_over_sampled_dataset(self):
        """
        Returns the fixed (as in "not moving") oversampled dataset.
        That means the oversampled samples from the smaller classes remain the same across all iterations.
        It is also a finite dataset.
        """
        class_datasets = self._make_class_datasets(subsample=False, seed=self.rng_seed)
        max_n_samples = max(self.n_samples_per_class)
        self.training_size = self.n_classes * floor(max_n_samples * (1 - self.val_split))

        fixed_over_sampled_dataset = class_datasets[0].take(floor(max_n_samples * (1 - self.val_split)))
        for ds in class_datasets[1:]:
            fixed_over_sampled_dataset = fixed_over_sampled_dataset.concatenate(
                ds.take(floor(max_n_samples * (1 - self.val_split))))

        if self.preprocessing_function is not None:
            fixed_over_sampled_dataset = fixed_over_sampled_dataset.map(self.preprocessing_function)

        return (fixed_over_sampled_dataset.
                shuffle(ceil(self.n_classes * max_n_samples * (1 - self.val_split))).
                batch(self.batch_size).
                prefetch(tf.data.AUTOTUNE))

    def get_fixed_sub_sampled_dataset(self):
        """
        Returns the fixed (as in "not moving") subsampled dataset.
        That means the subsample partition stays the same at every epoch.
        It is also a finite dataset.
        """
        class_datasets = self._make_class_datasets(subsample=True, seed=self.rng_seed)
        min_n_samples = min(self.n_samples_per_class)
        self.training_size = self.n_classes * floor(min_n_samples * (1 - self.val_split))

        fixed_sub_sampled_dataset = class_datasets[0].take(floor(min_n_samples * (1 - self.val_split)))
        for ds in class_datasets[1:]:
            fixed_sub_sampled_dataset = fixed_sub_sampled_dataset.concatenate(
                ds.take(floor(min_n_samples * (1 - self.val_split))))

        if self.preprocessing_function is not None:
            fixed_sub_sampled_dataset = fixed_sub_sampled_dataset.map(self.preprocessing_function)

        return (fixed_sub_sampled_dataset.
                shuffle(ceil(self.n_classes * min_n_samples * (1 - self.val_split))).
                batch(self.batch_size).
                prefetch(tf.data.AUTOTUNE))

    def get_data_tensor(self, oversample: bool = True):
        """
        Returns the full tensors (as np.ndarray) of the training and validation data, along with their labels
        You can choose to return the oversampled or the subsampled dataset.
        """
        n_training_samples_per_class = max(self.n_samples_per_class) if oversample else min(self.n_samples_per_class)
        n_training_samples_per_class = ceil(n_training_samples_per_class * (1 - self.val_split))

        n_validation_samples_per_class = max(self.n_samples_per_class) if oversample else min(self.n_samples_per_class)
        n_validation_samples_per_class = floor(n_training_samples_per_class * self.val_split)

        training_size = self.n_classes * n_training_samples_per_class
        validation_size = self.n_classes * n_validation_samples_per_class

        training = np.empty((training_size, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
        labels = np.empty(training_size, dtype=np.uint8)
        validationss = []
        val_labelss = []

        np.random.seed(self.rng_seed)

        cpt = 0
        for k in range(self.n_classes):
            class_files = glob.glob("{}/{}/*".format(self.data_dir, self.labels[k]))
            
            idx = np.arange(self.n_samples_per_class[k])
            np.random.shuffle(idx)

            n_available_training_samples = ceil((1 - self.val_split) * self.n_samples_per_class[k])
            n_validation_samples = self.n_samples_per_class[k] - n_available_training_samples

            train_idx = idx[:n_available_training_samples].copy()
            val_idx = idx[n_available_training_samples:]

            validation = np.empty((n_validation_samples, self.input_shape[0], self.input_shape[1], self.input_shape[2]))
            validation_labels = np.empty(n_validation_samples)

            for val_cpt, i in enumerate(val_idx):
                file = class_files[i]
                validation[val_cpt, :, :, :] = np.array(Image.open(file), dtype=np.uint8)
                validation_labels[val_cpt] = k

            validationss.append(validation)
            val_labelss.append(validation_labels)

            n_full = n_training_samples_per_class // n_available_training_samples
            n_res = n_training_samples_per_class % n_available_training_samples

            for i in range(n_full):
                for j in train_idx:
                    file = class_files[j]
                    training[cpt, :, :, :] = np.array(Image.open(file), dtype=np.uint8)
                    labels[cpt] = k
                    cpt += 1

            for i in np.random.choice(train_idx, n_res, replace=False):
                file = class_files[i]
                training[cpt, :, :, :] = np.array(Image.open(file), dtype=np.uint8)
                labels[cpt] = k
                cpt += 1

        self.training_size = cpt
        validations = np.concatenate(validationss, axis=0)
        val_labels = np.concatenate(val_labelss)

        return training, labels, (validations, val_labels)

    def get_validation_dataset(self, augmented: bool = False):
        """
        Returns the validation dataset as generated at object instantation
        """
        val_ds = self.val_ds
        if augmented:
            val_ds = val_ds.map(self.preprocessing_function)
        return val_ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
