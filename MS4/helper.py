import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import configuration
import logging
import os
import numpy as np


def create_pairs(image_path, image_name, correct=0, incorrect=1, simple=True, size=1000):
    logging.info("Sanity check")
    for path in image_path:
        if not os.path.isfile(path):
            logging.warning(path)
            exit(0)
    image_pairs_1 = np.array([])
    image_pairs_2 = np.array([])
    image_pairs_name = np.array([])
    pair_labels = np.array([])
    if simple:  # for larger datasets because the old pairing method is not enough
        image_names_unique, counts = np.unique(image_name, return_counts=True)
        logging.info(f"Names {len(image_name)}")
        logging.info(f"Unique names {len(image_names_unique)}")
        logging.info(f"name counts: {list(zip(image_names_unique, counts))}")
        with open(configuration.INFO_PATH, 'w+') as file:
            # Write text to the file
            file.write(f"Names {len(image_name)}\n")
            file.write(f"Unique names {len(image_names_unique)}\n")
            file.write(f"Name counts: {list(zip(image_names_unique, counts))}\n")
        idx = {name: np.where(image_name == name)[0] for name in image_names_unique}
        idx_neg = {name: np.where(image_name != name)[0] for name in image_names_unique}
        # loop over all images
        for current_image_path, current_image_name in tqdm(zip(image_path, image_name), total=len(image_name)):
            try:
                current_idx = idx[current_image_name]
            except IndexError:
                print(current_image_path, current_image_name)
                exit(0)
            actual_size = min(size, len(current_idx))
            # random image same name
            pos_index = np.random.choice(current_idx, size=actual_size, replace=False)
            pos_path = image_path[pos_index]
            pos_names = image_name[pos_index]

            image_pairs_1 = np.concatenate((image_pairs_1, np.full(actual_size, current_image_path)))
            image_pairs_2 = np.concatenate((image_pairs_2, pos_path))
            pair_labels = np.concatenate((pair_labels, np.ones(len(pos_path)) * correct))
            pos_names_stack = [f"{a}|{b}" for a, b in zip(np.full(len(pos_path), current_image_name), pos_names)]
            image_pairs_name = np.concatenate((image_pairs_name, pos_names_stack))
            # random image different identity
            # idx_neg = {name: np.where(image_name != name)[0] for name in image_names_unique}
            neg_index = np.random.choice(idx_neg[current_image_name], size=actual_size, replace=False)
            neg_path = image_path[neg_index]
            neg_names = image_name[neg_index]

            image_pairs_1 = np.concatenate((image_pairs_1, np.full(actual_size, current_image_path)))
            image_pairs_2 = np.concatenate((image_pairs_2, neg_path))
            pair_labels = np.concatenate((pair_labels, np.ones(len(neg_path)) * incorrect))
            neg_names_stack = [f"{a}|{b}" for a, b in zip(np.full(len(pos_path), current_image_name), neg_names)]
            image_pairs_name = np.concatenate((image_pairs_name, neg_names_stack))
    else:
        # WARNING: this method is dangerous to use, because it generates a very uneven dataset
        for i in tqdm(range(len(image_name))):
            for j in range(i + 1, len(image_name)):
                image_pairs_1.append(image_path[i])
                image_pairs_2.append(image_path[j])

                image_pairs_name.append([image_name[i], image_name[j]])
                pair_labels.append(correct if image_name[i] == image_name[j] else incorrect)

    print(len(image_pairs_1), len(image_pairs_2), len(image_pairs_name), len(pair_labels))
    # Create a DataFrame
    data = {'image1': image_pairs_1, 'image2': image_pairs_2, 'ImagePairsName': image_pairs_name,
            'PairLabels': pair_labels}
    df = pd.DataFrame(data)

    # Save to CSV
    logging.info("Saving to csv")
    df.to_csv(configuration.PAIR_PATH, index=False)


def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)
    return image


def load_images(pair_1, pair_2, label):
    image_1 = load_image(pair_1)
    image_2 = load_image(pair_2)
    return (image_1, image_2), label


def load_images_2(pair_1, pair_2):
    image_1 = load_image(pair_1)
    image_2 = load_image(pair_2)
    return (image_1, image_2)
