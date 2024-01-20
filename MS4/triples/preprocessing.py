import os.path
from itertools import product
import numpy as np
import pandas as pd
from tqdm import tqdm
import random


def make_promi_data(source_promis):
    names = []
    for image in os.listdir(source_promis):
        image_split = image.split("_")
        name = image_split[0]
        names.append(name)

    # make set
    names = set(names)
    # make numpy array
    names = np.array(list(names))
    print(names)
    parent_dir = '/home/sarah/Deep-Learning/MS4/data/triple_promis'
    for promi in names:
        path = os.path.join(parent_dir, promi)

        if not os.path.exists(path):
            os.mkdir(path)
            print(f"Ordner '{promi}' wurde erstellt.")
        else:
            print(f"Ordner '{promi}' existiert bereits.")

    image_dir = '/home/sarah/Deep-Learning/MS4/data/promis'
    triple_promis = '/home/sarah/Deep-Learning/MS4/data/triple_promis'
    for image in tqdm(os.listdir(image_dir)):
        image_path = os.path.join(image_dir, image)

        image_split = image.split("_")
        name = image_split[0]

        img = cv2.imread((image_path))
        destination = triple_promis + '/{}'.format(name)
        cv2.imwrite(os.path.join(destination, image), img)


def split_images_randomly(image_dir, split_ratio=0.5):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    random.shuffle(image_files)

    # Berechne die Anzahl der Bilder für jede Liste basierend auf dem Split-Verhältnis
    split_index = int(len(image_files) * split_ratio)

    # Teile die Bilder in zwei Listen auf
    list1 = image_files[:split_index]
    list2 = image_files[split_index:]

    return list1, list2


def create_cross_product(list1, list2):
    cross_product = list(product(list1, list2))
    return cross_product


def choose_random_image(image_dir):
    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]

    random_image = random.choice(image_files)
    random_image_path = os.path.join(image_dir, random_image)

    return random_image_path


def make_triples_positive(source_promis):
    df_triples_positive = pd.DataFrame(columns=['anchor', 'positive', 'name', 'label'])

    for folder in tqdm(os.listdir(source_promis)):
        subfolder_path = os.path.join(source_promis, folder)

        anchor, positive = split_images_randomly(subfolder_path)

        cross_product_list = create_cross_product(anchor, positive)


        for tuple in cross_product_list:
            name = [tuple[0].split('_')[0], tuple[1].split('_')[0]]
            df_triples_positive = df_triples_positive._append(
                {'anchor': os.path.join(subfolder_path, tuple[0]),
                 'positive': os.path.join(subfolder_path, tuple[1]),
                 'name': name,
                 'label': 1}, ignore_index=True)

    df_triples_positive.to_csv('/home/sarah/Deep-Learning/MS4/data/triple_positive.csv')


def make_triples_negative():
    df_triples_negative = pd.DataFrame(columns=['anchor', 'negative', 'name', 'label'])

    df_triples_postivie = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/triple_positive.csv', index_col=0)
    anchor = df_triples_postivie['anchor']
    for element in tqdm(anchor):
        name = element.split('/')
        name = name[4].split('_')[0]
        negative = choose_random_image('/home/sarah/Deep-Learning/MS4/data/MS3_rawData(2)/MS3_rawData/UTKFace')
        df_triples_negative = df_triples_negative._append({
            'anchor': element, 'negative': negative, 'name': [name, 'face'], 'label': 0
        }, ignore_index=True)

    df_triples_negative.to_csv('/home/sarah/Deep-Learning/MS4/data/triple_negative.csv')

