# image size 200x200
import cv2
import os
from os import listdir

# get the path/directory
path_out = "C:/Users/sarah/OneDrive/Desktop/Deep Learning Info&Daten/test"
path_in = "C:/Users/sarah/Deep-Learning/data/monkey"
i = 0

for images in os.listdir(path_in):
    # read image from folder
    input_image = os.path.join(path_in, images)
    img = cv2.imread(input_image)
    i = i + 1
    # crop image
    crop_img = img[0:200, 0:200]
    filename = "{}.jpg".format(i)

    # save image to new folder
    cv2.imwrite(os.path.join(path_out, filename), crop_img)
    cv2.waitKey(0)
