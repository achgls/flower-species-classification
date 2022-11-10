import os
import glob
from PIL import Image

data_dir = "../res/images"
image_files = glob.glob("../res/images/*/*")
total_n_images = image_files

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

n_images = dict()
for label in labels:
    n_images[label] = len(os.listdir("{}/{}/".format(data_dir, label)))

try:
    img = Image.open(image_files[0])
    img_width, img_height, img_mode = img.width, img.height, img.mode
    assert all([
        Image.open(image).width == img_width and
        Image.open(image).height == img_height and
        Image.open(image).mode == img_mode for
        image in glob.glob("../res/images/*/*")
    ])
    input_width, input_height, input_mode = img_width, img_height, img_mode
except AssertionError as err:
    print("Error: data unexpectedly has inconsistent size or color-encoding")
    raise AssertionError.with_traceback()

