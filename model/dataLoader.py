import pandas as pd
import tensorflow as tf
import numpy as np


def getTrainDataset(
    directory, seed=125, batch_size=32, size_img=(200, 200), validation=0.2
):
    """
    import Training Dataset from directory

    directory: path to directory
    seed: randomly shuffle vor validation split
    batch_size: size of data batches
    size_img: size of imported images
    validation: fraction of data to reserve for validation [0,1]
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="binary",
        class_names=["face", "noFace"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=size_img,
        shuffle=True,
        seed=seed,
        validation_split=validation,
        subset="both",
        crop_to_aspect_ratio=True,
    )


def getTestDataset(directory, batch_size=32, size_img=(200, 200)):
    """
    import Training Dataset from directory

    directory: path to directory
    batch_size: size of data batches
    size_img: size of imported images
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        label_mode="binary",
        class_names=["face", "noFace"],
        color_mode="rgb",
        batch_size=batch_size,
        image_size=size_img,
        shuffle=True,
        crop_to_aspect_ratio=True,
    )
