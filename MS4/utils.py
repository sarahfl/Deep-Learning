"""
https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configuration
import cv2


def load_image(image_path):
    """
    :param image_path: path to file
    :return: Image
    """
    image = cv2.imread(image_path, cv2.COLOR_BGR2RGB)  # , cv2.COLOR_BGR2RGB
    return image


def read_csv():
    df = pd.read_csv('MS4/preprocessing/promis.csv', index_col=0)  # df_face
    # df_no_face = pd.read_csv('MS4/preprocessing/noFace_regression.csv', index_col=0)
    # delta = len(df_no_face) - 12500
    # drop_indices = np.random.choice(df_no_face.index, delta, replace=False)
    # df_subset_noFace = df_no_face.drop(drop_indices)

    # df = pd.concat([df_face, df_no_face], axis=0, ignore_index=True)

    # shuffle dataframe
    # train_df = df.sample(frac=1)  # TODO: Inlude again
    return df  # train_df


def make_pairs(images, identities, num_identities, stop=False):
    pair_images = []
    pair_labels = []

    print(f"Number of unique identities {num_identities}")

    # f(identity) = list of all indexes of said identity
    idx = [np.where(identities == i)[0] for i in range(0, num_identities)]

    # loop over all images
    for index_a in range(len(images)):
        # current
        current_image = images[index_a]
        identity = identities[index_a]
        # random image same identity
        index_b = np.random.choice(idx[identity])
        pos_image = images[index_b]
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])

        # random image different identity
        neg_idx = np.where(identities != identity)[0]
        neg_image = images[np.random.choice(neg_idx)]
        pair_images.append([current_image, neg_image])
        pair_labels.append([0])

    # return [(image, image), ...], labels
    return np.array(pair_images), np.array(pair_labels)


def euclidean_distance(vectors):
    # unpack the vectors into separate lists
    (featsA, featsB) = vectors
    # compute the sum of squared distances between the vectors
    sum_squared = tf.keras.backend.sum(tf.keras.backend.square(featsA - featsB), axis=1,
                                       keepdims=True)
    # return the Euclidean distance between the vectors
    return tf.keras.backend.sqrt(tf.keras.backend.maximum(sum_squared, tf.keras.backend.epsilon()))


def plot_training(history, plot_path):
    # construct a plot that plots and saves the training history
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend(loc="lower left")
    plt.savefig(plot_path)


def display_image(x, y, image_index):
    image_index = 0  # Change this index to display different images
    # Display the image
    plt.imshow(x[image_index], cmap='gray')  # Display a grayscale image
    plt.title(f"Label: {y[image_index]}")  # Show the corresponding label
    plt.axis('off')  # Hide axis ticks and labels
    plt.show()
