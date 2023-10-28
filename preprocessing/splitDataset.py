import python_splitter
import os
import cv2
import tqdm
import numpy as np


def copyImagesToMS2Data():
    noFaceDirList = ["animals10", "landscape", "monkey", "natural-images"]
    faceDirList = ["UTKFace"]
    dir = "C:/Users/sarah/Deep-Learning/data/"
    path_out_face = "C:/Users/sarah/Deep-Learning/MS2Data/face"
    path_out_noFace = "C:/Users/sarah/Deep-Learning/MS2Data/noFace"

    for folder in noFaceDirList:
        path_to_folder = os.path.join(dir, folder)
        print(folder)
        for images in os.listdir(path_to_folder):
            # read image from folder
            input_image = os.path.join(path_to_folder, images)
            img = cv2.imread(input_image)
            image_name = folder + "_" + images
            # save image to new folder
            cv2.imwrite(os.path.join(path_out_noFace, image_name), img)
            cv2.waitKey(0)

    for folder in faceDirList:
        path_to_folder = os.path.join(dir, folder)
        print(folder)
        for images in os.listdir(path_to_folder):
            # read image from folder
            input_image = os.path.join(path_to_folder, images)
            img = cv2.imread(input_image)

            # save image to new folder
            cv2.imwrite(os.path.join(path_out_face, images), img)
            cv2.waitKey(0)


# copyImagesToMS2Data()
face = "C:/Users/sarah/Deep-Learning/MS2Data/face/"
noFace = "C:/Users/sarah/Deep-Learning/MS2Data/noFace/"
len_face = len(os.listdir(face))
len_noFace = len(os.listdir(noFace))
deltaFace = int(len_noFace) - int(len_face)
print("face", len_face)
print("noFace", len_noFace)
print("delta", deltaFace)


def deleteimages():
    noFace_dir = os.listdir(noFace)
    selected_image = np.random.choice(noFace_dir, deltaFace)
    i = 0
    e = 0
    for file in selected_image:
        try:
            os.remove(noFace + file)
            i = i + 1
            print(i)
        except:
            e = e + 1
            print("error finding picture")
    print("Missing: ", e)


# deleteimages()
# https://github.com/bharatadk/python_splitter
python_splitter.split_from_folder(
    "C:/Users/sarah/Deep-Learning/MS2Data", train=0.8, test=0.2
)
