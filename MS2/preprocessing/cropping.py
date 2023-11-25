import cv2
import os
from os import listdir
from tqdm import tqdm
from resizeImage import resizeAndPad

# get the path/directory
path_out = "/home/sarah/Desktop/Train_Test_Folder_2/noFace"
# path_in = "C:/Users/sarah/Deep-Learning/data/monkey"
path = "/home/sarah/Desktop/Train_Test_Folder_2/natural_images/natural_images"

# folder with sub-folders
for folder in tqdm(os.listdir(path)):
    i = 0
    foldername = folder
    path_to_folder = os.path.join(path, foldername)
    for images in tqdm(os.listdir(path_to_folder)):
        # read image from folder
        input_image = os.path.join(path_to_folder, images)
        img = cv2.imread(input_image)

        # crop image
        # cropped_img = resizeAndPad(img, (200, 200))

        # name new image
        i = i + 1
        filename = 'natural_images_' + foldername + "_{}.jpg".format(i)
        # save image to new folder
        cv2.imwrite(os.path.join(path_out, filename), img)
        cv2.waitKey(0)

'''# folder without sub-folder
i = 0
for images in tqdm(os.listdir(path)):
    # read image from folder
    input_image = os.path.join(path, images)
    img = cv2.imread(input_image)

    # crop image
    # cropped_img = resizeAndPad(img, (200, 200))

    # name new image
    i = i + 1
    filename = "landscape_{}.jpg".format(i)

    # save image to new folder
    cv2.imwrite(os.path.join(path_out, filename), img)
    cv2.waitKey(0)'''
