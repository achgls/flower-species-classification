import os
import glob
from math import log
from PIL import Image


# Data directory
data_dir = "../res/images"


# List of all image file paths
image_files = glob.glob("../res/images/*/*")


# Total number of images in the given dataset
total_n_images = len(image_files)


# The string labels corresponding to each class
labels = [
    "Species1",
    "Species2",
    "Species3",
    "Species4",
    "Species5",
    "Species6",
    "Species7",
    "Species8"
]


# Number of classes
n_species = len(labels)


# Number of images given for each distinct class
n_images: dict[str, int] = {}
for label in labels:
    n_images[label] = len(os.listdir("{}/{}/".format(data_dir, label)))


# Set the input data shape while asserting that it's consistent
try:
    img = Image.open(image_files[0])
    img_width, img_height, img_mode = img.width, img.height, img.mode
    assert all([
        Image.open(image).width == img_width and
        Image.open(image).height == img_height and
        Image.open(image).mode == img_mode for
        image in glob.glob("../res/images/*/*")
    ])
except AssertionError as err:
    print("Error: data unexpectedly has inconsistent size or color-encoding")
    raise AssertionError.with_traceback()

input_width, input_height, input_mode = img_width, img_height, img_mode
input_shape = (input_height, input_width, 3)


# Class weights for loss weighting (imbalanced classes)
class_loss_weights = [
    (1 / n_images[species]) * (total_n_images / n_species) for species in labels
]


# Initial biases for the output neurons
initial_class_biases = [
    log(n_images[species] / (total_n_images - n_images[species])) for species in labels
]