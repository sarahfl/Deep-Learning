import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Tensorflow Tutorial für Transfer learning
# https://www.tensorflow.org/tutorials/images/transfer_learning


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


def loadModelTraining(modelType, inputSize, dropout, classes=None):
    # noch eigene layer zu den Modellen hinzufügen?
    if modelType == "mobileNetV2Scratch":
        mobileNetV2Scratch = tf.keras.applications.MobileNetV2(
            input_shape=(inputSize, inputSize, 3),
            include_top=False,
            weights=None,
            dropout_rate=dropout,
            classifier_activation="sigmoid",
            include_preprocessing=True,
        )
        return mobileNetV2Scratch

    if modelType == "mobileNetV2":
        mobileNetV2 = tf.keras.applications.MobileNetV2(
            input_shape=(inputSize, inputSize, 3),
            weights="imagenet",
            classes=1000,
            dropout_rate=dropout,
            classifier_activation="sigmoid",
            include_preprocessing=True,
        )
        return mobileNetV2


def trainModel(model, trainingData, validationData, epochs, fileName):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    log_dir = "model/logsTensorBoard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    # mehr optionen möglich
    tensorBoard = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    model.summary()

    tf.debugging.set_log_device_placement(True)

    history = model.fit(
        trainingData,
        epochs=epochs,
        validation_data=validationData,
        callbacks=[tensorBoard],
    )

    # save model
    model.save("model/savedModels" + fileName)
    return history


def plotModel(history, plotName):
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label="Training Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.legend(loc="lower right")
    plt.ylabel("Accuracy")
    plt.ylim([min(plt.ylim()), 1])
    plt.title("Training and Validation Accuracy")

    plt.subplot(2, 1, 2)
    plt.plot(loss, label="Training Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.legend(loc="upper right")
    plt.ylabel("Cross Entropy")
    plt.ylim([0, 1.0])
    plt.title("Training and Validation Loss")
    plt.xlabel("epoch")

    plt.show()

    # save plot
    plt.savefig("model/plots/" + plotName)
