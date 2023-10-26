import cv2
import os
from os import listdir
from tqdm import tqdm
from resizeImage import resizeAndPad

# get the path/directory
path_out = "C:/Users/sarah/OneDrive/Desktop/Deep Learning Info&Daten/test"
path_in = "C:/Users/sarah/Deep-Learning/data/monkey"
i = 0

for images in tqdm(os.listdir(path_in)):
    # read image from folder
    input_image = os.path.join(path_in, images)
    img = cv2.imread(input_image)

    # crop image
    cropped_img = resizeAndPad(img, (200, 200))

    # name new image
    i = i + 1
    filename = "monkey_{}.jpg".format(i)

    # save image to new folder
    cv2.imwrite(os.path.join(path_out, filename), cropped_img)
    cv2.waitKey(0)
