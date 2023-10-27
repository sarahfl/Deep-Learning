import cv2
import os
from os import listdir
from tqdm import tqdm
from resizeImage import resizeAndPad

# get the path/directory
path_out = "C:/Users/sarah/Deep-Learning/data/animals10"
# path_in = "C:/Users/sarah/Deep-Learning/data/monkey"
path = "C:/Users/sarah/OneDrive/Desktop/Deep Learning Info&Daten/animals-10/raw-img/"


for folder in tqdm(os.listdir(path)):
    i = 0
    foldername = folder
    path_to_folder = os.path.join(path, foldername)
    for images in os.listdir(path_to_folder):
        # read image from folder
        input_image = os.path.join(path_to_folder, images)
        img = cv2.imread(input_image)

        # crop image
        cropped_img = resizeAndPad(img, (200, 200))

        # name new image
        i = i + 1
        filename = foldername + "_{}.jpg".format(i)

        # save image to new folder
        cv2.imwrite(os.path.join(path_out, filename), cropped_img)
        cv2.waitKey(0)
