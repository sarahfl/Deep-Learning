import pandas as pd
from tqdm import tqdm
import tensorflow as tf


def create_pairs(image_path, image_name):
    image_pairs_1 = []
    image_pairs_2 = []
    image_pairs_name = []
    pair_labels = []
    for i in tqdm(range(len(image_name))):
        for j in range(i + 1, len(image_name)):
            image_pairs_1.append(image_path[i])
            image_pairs_2.append(image_path[j])


            # image_pairs_path.append([p+ image_path[i], p+ image_path[j]])
            image_pairs_name.append([image_name[i], image_name[j]])

            if image_name[i] == image_name[j]:
                pair_labels.append(1)
            else:
                pair_labels.append(0)

    # Create a DataFrame
    data = {'image1': image_pairs_1, 'image2': image_pairs_2, 'ImagePairsName': image_pairs_name,
            'PairLabels': pair_labels}
    df = pd.DataFrame(data)

    # Save to CSV
    df.to_csv('MS4/data/pairs.csv', index=False)


def load_image(file_path):
    image = tf.io.read_file(file_path)
    image = tf.image.decode_image(image, channels=3)

    return image


def load_images(pair_1, pair_2, label):
    image_1 = load_image(pair_1)
    image_2 = load_image(pair_2)
    return (image_1, image_2), label