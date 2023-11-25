import cv2
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

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

def sexDistribution():
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
