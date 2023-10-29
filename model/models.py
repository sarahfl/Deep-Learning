import pandas as pd
import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt

# Tensorflow Tutorial für Transfer learning
# https://www.tensorflow.org/tutorials/images/transfer_learning


def get_train_dataset(
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


def get_test_dataset(directory, batch_size=32, size_img=(200, 200)):
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


def load_model_training(model_type, image_size_quadratic, dropout, classes=None):
    if model_type == "mobileNetV2Scratch":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_size_quadratic, image_size_quadratic, 3),
            include_top=False,
            weights=None,
            classifier_activation="sigmoid",
        )
    else:  # model_type == "mobileNetV2":
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=(image_size_quadratic, image_size_quadratic, 3),
            weights="imagenet",
            classes=1000,
            classifier_activation="sigmoid",
        )
    base_model.trainable = False
    return base_model


def data_augmentation(x):
    return x


def train_model(model, training_data, validation_data, epochs, file_name):
    training_data = training_data.prefetch(buffer_size=tf.data.AUTOTUNE)
    validation_data = validation_data.prefetch(buffer_size=tf.data.AUTOTUNE)

    # TODO: make prettier
    # TODO: Check why loss is not changing at all.
    # https://www.tensorflow.org/tutorials/images/transfer_learning#create_the_base_model_from_the_pre-trained_convnets
    image_batch, label_batch = next(iter(training_data))
    feature_batch = model(image_batch)
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    feature_batch_average = global_average_layer(feature_batch)
    print(feature_batch_average.shape)
    prediction_layer = tf.keras.layers.Dense(1)
    prediction_batch = prediction_layer(feature_batch_average)
    print(prediction_batch.shape)
    inputs = tf.keras.Input(shape=(200, 200, 3))
    x = data_augmentation(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    x = model(x, training=False)
    x = global_average_layer(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    tf.keras.utils.plot_model(model, show_shapes=True)
    model = tf.keras.Model(inputs, outputs)
    len(model.trainable_variables)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=["accuracy"],  # [tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')]
    )
    log_dir = "model/logsTensorBoard/" + datetime.datetime.now().strftime(
        "%Y%m%d-%H%M%S"
    )
    # mehr optionen möglich
    tensor_board = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    # model.summary()

    tf.debugging.set_log_device_placement(True)
    print(training_data)
    history = model.fit(
        training_data,
        epochs=epochs,
        validation_data=validation_data,
        callbacks=[tensor_board],
    )

    # save model
    model.save("model/savedModels" + file_name)
    return history


def plot_model(history, plot_name):
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
    plt.savefig("model/plots/" + plot_name)
