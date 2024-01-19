import pandas as pd
from tqdm import tqdm
import tensorflow as tf
import configuration
import logging
import os
import numpy as np


def create_pairs(image_path, image_name, invers=False, simple=False, size=10):
    logging.info("Sanity check")
    for path in image_path:
        if not os.path.isfile(path):
            logging.warning(path)
            exit(0)
    image_pairs_1 = []
    image_pairs_2 = []
    image_pairs_name = []
    pair_labels = []
    if simple:  # for larger datasets because the old pairing method is not enough
        image_names_unique, counts = np.unique(image_name, return_counts=True)
        logging.info(f"Names {len(image_name)}")
        logging.info(f"Unique names {len(image_names_unique)}")
        logging.info(f"name counts: {list(zip(image_names_unique, counts))}")
        idx = {name: np.where(image_name == name)[0] for name in image_names_unique}
        # loop over all images
        for current_image_path, current_image_name in tqdm(zip(image_path, image_name), total=len(image_name)):
            try:
                current_idx = idx[current_image_name]
            except IndexError:
                print(current_image_path, current_image_name)
                exit(0)
            # random image same name
            pos_index = np.random.choice(current_idx, size=min(size, len(current_idx)), replace=False)
            pos_path = image_path[pos_index]
            pos_name = image_name[pos_index]

            image_pairs_1 += current_image_path
            image_pairs_2 += pos_path
            pair_labels += [1]*len(pos_path)
            image_pairs_name += ["/"]*len(pos_path)
            # random image different identity
            name_complement = np.setdiff1d(image_name, current_idx)
            neg_index = np.random.choice(name_complement, size=min(size, len(name_complement)), replace=False)
            neg_path = image_path[neg_index]
            neg_name = image_name[neg_index]

            image_pairs_1 += current_image_path
            image_pairs_2 += neg_path
            pair_labels += [0]*len(neg_path)
            image_pairs_name += ["/"]*len(neg_path)
    else:
        for i in tqdm(range(len(image_name))):
            for j in range(i + 1, len(image_name)):
                image_pairs_1.append(image_path[i])
                image_pairs_2.append(image_path[j])

                image_pairs_name.append([image_name[i], image_name[j]])

                pair_labels.append(image_name[i] == image_name[j] if invers else image_name[i] != image_name[j])

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
