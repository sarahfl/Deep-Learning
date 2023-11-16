import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image
from PIL.Image import Resampling
from tqdm import tqdm

"""
Resize Image and add padding if needed.
Source: https://stackoverflow.com/questions/44720580/resize-image-to-maintain-aspect-ratio-in-python-opencv
size: requires a tupel e.g. (200,200)
"""

path_in = "/home/sarah/Desktop/Train_Test_Folder_2/noFace"
path_out = "/home/sarah/Desktop/Train_Test_Folder_2/noFace_scale"


def resizePIL(image_url):
    with Image.open(image_url) as im:
        # Provide the target width and height of the image
        (width, height) = (200, 200)
        im_resized = im.resize((width, height), resample=Resampling.LANCZOS)
        return im_resized


def resizeAndPad(img, size, padColor=255):
    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw:  # shrinking image
        interp = cv2.INTER_AREA

    else:  # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = float(w) / h
    saspect = float(sw) / sh

    if (saspect >= aspect) or (
            (saspect == 1) and (aspect <= 1)
    ):  # new horizontal image
        new_h = sh
        new_w = np.round(new_h * aspect).astype(int)
        pad_horz = float(sw - new_w) / 2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(
            int
        )
        pad_top, pad_bot = 0, 0

    elif (saspect < aspect) or ((saspect == 1) and (aspect >= 1)):  # new vertical image
        new_w = sw
        new_h = np.round(float(new_w) / aspect).astype(int)
        pad_vert = float(sh - new_h) / 2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0

    # set pad color
    if len(img.shape) == 3 and not isinstance(
            padColor, (list, tuple, np.ndarray)
    ):  # color image but only one color provided
        padColor = [padColor] * 3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(
        scaled_img,
        pad_top,
        pad_bot,
        pad_left,
        pad_right,
        borderType=cv2.BORDER_CONSTANT,
        value=padColor,
    )

    return scaled_img


i = 0
for image in tqdm(os.listdir(path_in)):
    input_image = os.path.join(path_in, image)
    resizedImage = resizePIL(input_image)
    resizedImage.save(os.path.join(path_out, image))
