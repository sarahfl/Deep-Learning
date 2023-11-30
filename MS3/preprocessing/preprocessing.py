import cv2
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def classDistribution():
    df = pd.DataFrame(columns=['train_folder', 'count'])
    parent_folder = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/train/"
    for folder in tqdm(os.listdir(parent_folder)):
        folder_path = os.path.join(parent_folder, folder)

        image_count = sum(1 for image in os.listdir(folder_path))

        print(image_count)
        df = df._append({'train_folder': folder, 'count': image_count}, ignore_index=True)

    print(df)
    # df.to_csv('val_classDistribution.csv', sep='\t')

    # Bar-Plot erstellen

    plt.bar(df['train_folder'], df['count'])
    plt.title('Verteilung der Bilder über die 10 Klassen')
    # Achsentitel hinzufügen
    plt.xlabel('Klasse')
    plt.ylabel('Anzahl der Bilder')

    # Diagramm anzeigen

    plt.savefig('train_classDistribution.png')
    plt.show()


def copyImageas():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/noFace"
    path_out = "/home/sarah/Deep-Learning/MS3/preprocessing/Data"

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



def createFaceCSV_classification():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/UTKFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        # split image name
        image_split = image.split("_")

        age = int(image_split[0])
        gender = int(image_split[1])

        result = ''
        if 1 <= age <= 2:
            result = 'age0'
        elif 3 <= age <= 9:
            result = 'age1'
        elif 10 <= age <= 20:
            result = 'age2'
        elif 21 <= age <= 27:
            result = 'age3'
        elif 28 <= age <= 45:
            result = 'age4'
        elif 46 <= age <= 65:
            result = 'age5'
        elif 66 <= age <= 116:
            result = 'age6'

        df = df._append({'path': input_image, 'age': result, 'gender': gender, 'face': 0}, ignore_index=True)

    df.to_csv('Data/Face_classification.csv')


def createNoFaceCSV_classification():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/noFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        df = df._append({'path': input_image, 'age': 'age7', 'gender': 3, 'face': 1}, ignore_index=True)

    df.to_csv('Data/noFace_classification.csv')


def createFaceCSV_regression():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/UTKFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        # split image name
        image_split = image.split("_")

        age = int(image_split[0])
        gender = int(image_split[1])

        df = df._append({'path': input_image, 'age': age, 'gender': gender, 'face': 0}, ignore_index=True)

    df.to_csv('Data/Face_regression.csv')

def createNoFaceCSV_regression():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/noFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        df = df._append({'path': input_image, 'age': '0', 'gender': 3, 'face': 1}, ignore_index=True)

    df.to_csv('Data/noFace_regression.csv')


createFaceCSV_regression()
createNoFaceCSV_regression()