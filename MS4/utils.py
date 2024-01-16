import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import configuration
import cv2


def load_and_preprocess_image(image_path):
    """
    :param image_path: path to file
    :return: Image
    """
    image = cv2.imread(image_path)  # , cv2.COLOR_BGR2RGB
    image = image / 255.0  # scale [0-1]
    image = np.expand_dims(image, axis=-1)
    return image


def read_csv():
    df = pd.read_csv('MS4/preprocessing/promis.csv', index_col=0)  # df_face
    # df_no_face = pd.read_csv('MS4/preprocessing/noFace_regression.csv', index_col=0)
    # delta = len(df_no_face) - 12500
    # drop_indices = np.random.choice(df_no_face.index, delta, replace=False)
    # df_subset_noFace = df_no_face.drop(drop_indices)

    # df = pd.concat([df_face, df_no_face], axis=0, ignore_index=True)

    # shuffle dataframe
    train_df = df.sample(frac=1)
    return train_df


def make_pairs(images, labels):
    # initialize two empty lists to hold the (image, image) pairs and
    # labels to indicate if a pair is positive or negative
    pair_images = []
    pair_labels = []
    # calculate the total number of classes present in the dataset
    # and then build a list of indexes for each class label that
    # provides the indexes for all examples with a given label
    num_classes = len(np.unique(labels))
    idx = [np.where(labels == i)[0] for i in range(0, num_classes)]
    # loop over all images
    for idxA in range(len(images)):
        # grab the current image and label belonging to the current
        # iteration
        current_image = images[idxA]
        label = labels[idxA]
        # randomly pick an image that belongs to the *same* class
        # label
        idx_b = np.random.choice(idx[label])
        pos_image = images[idx_b]
        # prepare a positive pair and update the images and labels
        # lists, respectively
        pair_images.append([current_image, pos_image])
        pair_labels.append([1])
        # grab the indices for each of the class labels *not* equal to
        # the current label and randomly pick an image corresponding
        # to a label *not* equal to the current label
        neg_idx = np.where(labels != label)[0]
        neg_image = images[np.random.choice(neg_idx)]
        # prepare a negative pair of images and update our lists
        pair_images.append([current_image, neg_image])
        pair_labels.append([0])
    # return a 2-tuple of our image pairs and labels
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
