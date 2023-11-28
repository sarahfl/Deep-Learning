import cv2
import os
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import python_splitter
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


def splitTrainValTest():
    # https://github.com/bharatadk/python_splitter
    python_splitter.split_from_folder(
        "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData", train=0.8, test=0.1, val=0.1)


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


def createGenderSplit():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/UTKFace"
    path_out = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val/"

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)
        # read image
        img = cv2.imread(input_image)

        # split image name
        image_split = image.split("_")

        # male = 0
        # female = 1
        gender = int(image_split[1])

        if gender == 0:
            path = os.path.join(path_out, 'male')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif gender == 1:
            path = os.path.join(path_out, 'female')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)


def createSubfolders():
    parent_folder = "/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/train"
    subfolder_names = ["male", "female", "age0", "age1", "age2", "age3", "age4", "age5", "age6"]
    for subfolder_name in subfolder_names:
        subfolder_path = os.path.join(parent_folder, subfolder_name)
        os.makedirs(subfolder_path)
        print(f"Unterordner '{subfolder_name}' wurde erstellt in '{parent_folder}'.")


def createAgeGenderSplit():
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

        # male = 0
        # female = 1
        gender = int(image_split[1])

        if 1 <= age <= 2 and gender == 0:
            path = os.path.join(path_out, 'male_age0')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 1 <= age <= 2 and gender == 1:
            path = os.path.join(path_out, 'female_age0')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 3 <= age <= 9 and gender == 0:
            path = os.path.join(path_out, 'male_age1')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 3 <= age <= 9 and gender == 1:
            path = os.path.join(path_out, 'female_age1')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 10 <= age <= 20 and gender == 0:
            path = os.path.join(path_out, 'male_age2')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 10 <= age <= 20 and gender == 1:
            path = os.path.join(path_out, 'female_age2')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 21 <= age <= 27 and gender == 0:
            path = os.path.join(path_out, 'male_age3')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 21 <= age <= 27 and gender == 1:
            path = os.path.join(path_out, 'female_age3')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 28 <= age <= 45 and gender == 0:
            path = os.path.join(path_out, 'male_age4')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 28 <= age <= 45 and gender == 1:
            path = os.path.join(path_out, 'female_age4')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 46 <= age <= 65 and gender == 0:
            path = os.path.join(path_out, 'male_age5')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 46 <= age <= 65 and gender == 1:
            path = os.path.join(path_out, 'female_age5')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 66 <= age <= 116 and gender == 0:
            path = os.path.join(path_out, 'male_age6')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)
        elif 66 <= age <= 116 and gender == 1:
            path = os.path.join(path_out, 'female_age6')
            cv2.imwrite(os.path.join(path, image), img)
            cv2.waitKey(0)


def createDataFrameFace():
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

    df.to_csv('UTKFace.csv', sep='\t')

def createDataFramaNoFace():
    # get the path/directory
    path_in = "/home/sarah/Deep-Learning/MS3/preprocessing/MS3_rawData/noFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        df = df._append({'path': input_image, 'age': 'age7', 'gender': 3, 'face': 1}, ignore_index=True)

    df.to_csv('noFace.csv', sep='\t')

