import os
import random
import cv2
from tqdm import tqdm
from PIL import Image
from PIL.Image import Resampling
import pandas as pd


def resize_pil(image_url, width=200, height=200):
    with Image.open(image_url) as im:
        # Provide the target width and height of the image
        im_resized = im.resize((width, height), resample=Resampling.LANCZOS)
        return im_resized


def copy_images(source, destination):
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


def take_sample_images(source, destination, count):
    images = [f for f in os.listdir(source) if os.path.isfile(os.path.join(source, f))]

    random_images = random.sample(images, count)
    i = 0
    for image in random_images:
        image_source = os.path.join(source, image)
        img = cv2.imread(image_source)
        filename = 'face_{}.jpg'.format(i)
        cv2.imwrite(os.path.join(destination, filename), img)
        i = i + 1


def images_to_csv(source_promis, source_face, source_celeb_a, destination_csv):
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

    # celebA
    df_celeb_a = pd.read_csv(os.path.join(SOURCES_CELEB_A, "Anno", "identity_CelebA.txt"), sep=" ", names=["path", "name"])
    df_celeb_a["path"] = df_celeb_a["path"].apply(
        lambda file_name: os.path.join(SOURCES_CELEB_A, "img_align_celeba", file_name))

    df_celeb_a.to_csv(destination_csv + '/celeb_a.csv')


SOURCE_PROMIS = 'MS4/data/promis'
SOURCES_FACES = 'MS4/data/faces'
SOURCES_CELEB_A = 'MS4/data/CelebA/'  # 178, 218
DESTINATION_CSV = 'MS4/preprocessing'

images_to_csv(SOURCE_PROMIS, SOURCES_FACES, SOURCES_CELEB_A, DESTINATION_CSV)
