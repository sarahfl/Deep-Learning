##
import cv2
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from PIL.Image import Resampling


##

def promisToCSV_classification():
    path_to_prediction = '/home/sarah/Deep-Learning/MS3/prediction/deutschePromis'
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])
    for image in os.listdir(path_to_prediction):
        path = os.path.join(path_to_prediction, image)

        image_split = image.split("_")
        age = int(image_split[0])
        gender = int(image_split[1])
        name = image_split[2]

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
        elif age == 0:
            result = 'age7'

        face = 0
        if name == 'elmo.jpg':
            face = 1
        else:
            face = 0

        df = df._append({'path': path, 'age': result, 'gender': gender, 'face': face}, ignore_index=True)
    df.to_csv('/home/sarah/Deep-Learning/MS3/prediction/deutschePromis_classification.csv')


def promisToCSV_regression():
    path_to_prediction = 'MS3/Model/data/deutschePromis'
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])
    for image in os.listdir(path_to_prediction):
        path = os.path.join(path_to_prediction, image)

        image_split = image.split("_")
        age = int(image_split[0])
        gender = int(image_split[1])
        name = image_split[2]

        face = 0
        if name == 'elmo.jpg':
            face = 1
        else:
            face = 0

        df = df._append({'path': path, 'age': age, 'gender': gender, 'face': face}, ignore_index=True)
    df.to_csv('MS3/prediction/deutschePromis_regression.csv')


def starWarsToCSV_regression():
    path_to_prediction = 'MS3/Model/data/starWars'
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])
    for image in os.listdir(path_to_prediction):
        path = os.path.join(path_to_prediction, image)

        image_split = image.split("_")
        age = int(image_split[0])
        gender = int(image_split[1])
        name = image_split[2]

        face = 0
        if name == 'elmo.jpg':
            face = 1
        else:
            face = 0

        df = df._append({'path': path, 'age': age, 'gender': gender, 'face': face}, ignore_index=True)
    df.to_csv('MS3/prediction/starWars_regression.csv')


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
    df_face = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/Face_classification.csv', index_col=0)
    df_noFace = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/noFace_classification.csv', index_col=0)
    df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

    df = df['age'].to_list()
    age0 = df.count('age0')
    age1 = df.count('age1')
    age2 = df.count('age2')
    age3 = df.count('age3')
    age4 = df.count('age4')
    age5 = df.count('age5')
    age6 = df.count('age6')
    age7 = df.count('age7')

    age = ['1-2', '3-9', '10-20', '21-27', '28-45', '46-65', '66-116', 'no age']
    counts = [age0, age1, age2, age3, age4, age5, age6, age7]
    fontsize = 20

    plt.figure(figsize=(40, 18))
    bar_colors = ['tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:blue', 'tab:orange']
    plt.bar(age, counts, color=bar_colors)
    plt.title("Distribution of Age in UTKFace", fontsize=fontsize)
    plt.xlabel("Age", fontsize=fontsize)
    plt.ylabel("Number of Images", fontsize=fontsize)

    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    plt.savefig('plots/age_distribution_classification.png')
    plt.show()


def genderDistribution():
    df_face = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/Face_classification.csv', index_col=0)
    df_noFace = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/noFace_classification.csv', index_col=0)
    df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

    df = df['gender'].to_list()
    male = df.count(0)
    female = df.count(1)
    no_gender = df.count(3)

    gender = ['male', 'female', 'no gender']
    counts = [male, female, no_gender]
    fontsize = 20

    plt.figure(figsize=(20, 20))
    fig, ax = plt.subplots()
    bar_colors = ['tab:blue', 'tab:blue', 'tab:orange']
    ax.bar(gender, counts, color=bar_colors)

    ax.set_ylabel('Number of Images')
    ax.set_title('Distribution of gender')

    plt.savefig('plots/gender_distribution.png')

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

        df = df._append({'path': input_image, 'age': 'age7', 'gender': 2, 'face': 1}, ignore_index=True)

    df.to_csv('Data/noFace_classification.csv')


##
def createFaceCSV_regression():
    # get the path/directory
    path_in = "MS3/Model/data/UTKFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        # split image name
        image_split = image.split("_")

        age = int(image_split[0])
        gender = int(image_split[1])

        df = df._append({'path': input_image, 'age': age, 'gender': gender, 'face': 0}, ignore_index=True)

    df.to_csv('MS3/Model/data/Face_regression.csv')


##
def createNoFaceCSV_regression():
    # get the path/directory
    path_in = "MS3/Model/data/noFace"
    df = pd.DataFrame(columns=['path', 'age', 'gender', 'face'])

    for image in tqdm(os.listdir(path_in)):
        input_image = os.path.join(path_in, image)

        df = df._append({'path': input_image, 'age': '0', 'gender': 2, 'face': 1}, ignore_index=True)

    df.to_csv('MS3/Model/data/noFace_regression.csv')


##

def resizePIL(image_url):
    with Image.open(image_url) as im:
        # Provide the target width and height of the image
        (width, height) = (200, 200)
        im_resized = im.resize((width, height), resample=Resampling.LANCZOS)
        return im_resized


'''path_in = "/home/sarah/Pictures/test"
path_out = "/home/sarah/Deep-Learning/MS3/prediction/deutschePromis"
i = 0
for image in tqdm(os.listdir(path_in)):
    input_image = os.path.join(path_in, image)
    resizedImage = resizePIL(input_image)
    resizedImage.save(os.path.join(path_out, image))'''
