import cv2
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import python_splitter


def renameImages():
    # get the path/directory
    path_in = "/home/sarah/Desktop/UTKFace"
    path_out = "/home/sarah/Desktop/UTKFace_2"

    i = 0
    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)
        # read image
        img = cv2.imread(input_image)

        # split image name
        image_split = image.split("_")

        i = i + 1
        filename = "{age}_{sex}_face_{id}.jpg".format(age=image_split[0], sex=image_split[1], id=i)

        # save image to new folder
        cv2.imwrite(os.path.join(path_out, filename), img)
        cv2.waitKey(0)


def copyImageas():
    # get the path/directory
    path_in = "/home/sarah/Desktop/Train_Test_Folder_2/train/noFace"
    path_out = "/home/sarah/Desktop/noFace"

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)
        # read image
        img = cv2.imread(input_image)

        # save image to new folder
        cv2.imwrite(os.path.join(path_out, image), img)
        cv2.waitKey(0)


def ageDistribution():
    path_in = "/home/sarah/Desktop/UTKFace_2"

    arr = []
    for image in tqdm(os.listdir(path_in)):
        # split image name
        image_split = image.split("_")

        arr.append(int(image_split[0]))

    # count occurrence of age
    sorted_arr = np.sort(arr)
    result = Counter(sorted_arr)
    dict(result)
    dict(sorted(result.items()))

    age = result.keys()
    counts = result.values()
    fontsize = 20

    plt.figure(figsize=(40, 18))
    plt.bar(age, counts)
    plt.title("Distribution of Age in UTKFace", fontsize=fontsize)
    plt.xlabel("Age", fontsize=fontsize)
    plt.ylabel("Number of Images", fontsize=fontsize)
    dim = np.arange(1, 117, 1)
    plt.xticks(dim, fontsize=fontsize, rotation=90)
    plt.yticks(fontsize=fontsize)

    plt.savefig('age_distribution_utkface.png')
    plt.show()


def genderDistribution():
    path_in = "/home/sarah/Desktop/UTKFace_2"

    arr = []
    for image in tqdm(os.listdir(path_in)):
        # split image name
        image_split = image.split("_")

        arr.append(int(image_split[1]))

    # count occurrence of age
    sorted_arr = np.sort(arr)
    result = Counter(sorted_arr)
    dict(result)
    dict(sorted(result.items()))

    gender = ['male', 'female']
    counts = result.values()
    fig, ax = plt.subplots()
    bar_colors = ['tab:blue', 'tab:red']
    ax.bar(gender, counts, color=bar_colors)

    ax.set_ylabel('Number of Images')
    ax.set_title('Distribution of gender')

    plt.savefig('gender_distribution_utkface.png')

    plt.show()


def splitTrainValTest():
    # https://github.com/bharatadk/python_splitter
    python_splitter.split_from_folder(
        "/home/sarah/Desktop/MS3_Train_Val_Test", train=0.8, test=0.1, val=0.1)


def createGenderSplit():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/UTKFace"
    path_out_male = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/male"
    path_out_female = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/female"

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)
        # read image
        img = cv2.imread(input_image)

        # split image name
        image_split = image.split("_")

        if (image_split[1] == '0'):  # if male
            # save image to new folder
            cv2.imwrite(os.path.join(path_out_male, image), img)
            cv2.waitKey(0)
        elif (image_split[1] == '1'):  # if female
            cv2.imwrite(os.path.join(path_out_female, image), img)
            cv2.waitKey(0)


def createAgeSplit():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/UTKFace"
    path_out = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/"

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)
        # read image
        img = cv2.imread(input_image)

        # split image name
        image_split = image.split("_")

        # age0: 1 - 2 Jahre
        # age1: 3 - 9 Jahre
        # age2: 10 - 20 Jahre
        # age3: 21 - 27 Jahre
        # age4: 28 - 45 Jahre
        # age5: 46 - 65 Jahre
        # age6: 66 - 116 Jahre

        age = int(image_split[0])

        if 1 <= age <= 2:
            path = os.path.join(path_out, 'age0')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 3 <= age <= 9:
            path = os.path.join(path_out, 'age1')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 10 <= age <= 20:
            path = os.path.join(path_out, 'age2')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 21 <= age <= 27:
            path = os.path.join(path_out, 'age3')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 28 <= age <= 45:
            path = os.path.join(path_out, 'age4')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 46 <= age <= 65:
            path = os.path.join(path_out, 'age5')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 66 <= age <= 116:
            path = os.path.join(path_out, 'age6')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)

createAgeSplit()
