import glob
import os
from math import ceil

import tensorflow as tf
from PIL import Image


class BalancedDataPipe:
    def __init__(
            self,
            data_directory: str,
            rng_seed: int = 42,
            val_split: float = 0.15
    ):
        self.rng_seed = rng_seed
        self.data_dir = data_directory
        self.val_split = val_split

        image_files = glob.glob("{}/*/*".format(self.data_dir))
        self.total_n_samples = len(image_files)

        test_image = Image.open(image_files[0])
        img_height, img_width, img_mode = test_image.height, test_image.width, test_image.mode
        assert all(
            [
                Image.open(image).width == img_width and
                Image.open(image).height == img_height and
                Image.open(image).mode == img_mode for
                image in image_files
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
            seed=self.rng_seed,
            image_size=(96, 96),
            batch_size=None)  # Retrieve unbatched data

        self.val_ds = tf.keras.utils.image_dataset_from_directory(
            self.data_dir,
            validation_split=self.val_split,
            subset="validation",
            seed=self.rng_seed,
            image_size=(96, 96),
            batch_size=None)  # Retrieve unbatched data

    def _make_class_datasets(self, subsample: bool = False):
        train_datasets = []
        max_n_samples = max(self.n_samples_per_class)

        for class_number in range(self.n_classes):
            n_samples = self.n_samples_per_class[class_number]
            repeat_factor = ceil(max_n_samples / n_samples)
            class_dataset = (
                self.train_ds
                .filter(lambda features, label: label == class_number)
                .shuffle(n_samples, reshuffle_each_iteration=True)
            )
            if subsample is False:
                class_dataset = class_dataset.repeat(repeat_factor)

            train_datasets.append(class_dataset)

        return train_datasets

    def get_over_sampled_dataset(self):
        class_datasets = self._make_class_datasets(subsample=False)
        over_sampled_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets,
            stop_on_empty_dataset=True,
            weights=None,  # None = equally distributed
            seed=None  # Different sampling at every epoch (problem: inconsistent dataset size from epoch to epoch)
        )
        return over_sampled_ds

    def get_sub_sampled_dataset(self):
        class_datasets = self._make_class_datasets(subsample=True)
        sub_sampled_ds = tf.data.Dataset.sample_from_datasets(
            class_datasets,
            stop_on_empty_dataset=True,
            weights=None,  # None = equally distributed
            seed=None  # Different sampling at every epoch (problem: inconsistent dataset size from epoch to epoch)
        )
        return sub_sampled_ds

    def get_validation_dataset(self, batch_size: int = 32):
        return self.val_ds.batch(batch_size)
