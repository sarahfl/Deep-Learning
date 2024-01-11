import os
import random
import cv2
from tqdm import tqdm
from PIL import Image
from PIL.Image import Resampling
import pandas as pd


def resizePIL(image_url):
    with Image.open(image_url) as im:
        # Provide the target width and height of the image
        (width, height) = (200, 200)
        im_resized = im.resize((width, height), resample=Resampling.LANCZOS)
        return im_resized


def copyImages(source, destination):
    i = 0
    for image in tqdm(os.listdir(source)):
        input_image = os.path.join(source, image)
        # read image
        img = cv2.imread(input_image)

        # split image name
        image_split = image.split("_")
        age = int(image_split[0])
        if 16 < age < 80:
            # save image to new folder
            cv2.imwrite(os.path.join(destination, image), img)
            cv2.waitKey(0)


def takeSampleImages(source, destination, count):
    images = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    random_images = random.sample(images, count)
    i = 0
    for image in random_images:
        image_source = os.path.join(source, image)
        img = cv2.imread(image_source)
        filename = 'face_{}.jpg'.format(i)
        cv2.imwrite(os.path.join(destination, filename), img)
        i = i + 1


def imagesToCSV(source_promis, source_face, destination_csv):
    # promis
    df_promis = pd.DataFrame(columns=['path', 'name'])
    for image in os.listdir(source_promis):
        path = os.path.join(source_promis, image)

        image_split = image.split("_")
        name = image_split[0]
        df_promis = df_promis._append({'path': path, 'name': name}, ignore_index=True)

    df_promis.to_csv(destination_csv + '/promis.csv')

    # faces
    df_faces = pd.DataFrame(columns=['path', 'name'])
    for image in os.listdir(source_face):
        path = os.path.join(source_face, image)

        image_split = image.split("_")
        name = image_split[0]
        df_faces = df_faces._append({'path': path, 'name': name}, ignore_index=True)

    df_faces.to_csv(destination_csv + '/faces.csv')


source_promis = '/home/sarah/Desktop/promis'
source_faces = '/home/sarah/Desktop/faces'
destination_csv = '/home/sarah/Deep-Learning/MS4/preprocessing'

imagesToCSV(source_promis, source_faces, destination_csv)
