from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_siamese_model(input_shape, embedding_dim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(input_shape)

    base_model = MobileNetV2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,
        include_top=False,
    )
    data_augmentation = Sequential([
        RandomFlip('horizontal'),
        RandomRotation(0.2),
    ])

    # first
    x = data_augmentation(inputs)
    x = preprocess_input(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280, activation='relu')(x)
    x = Dropout(0.5)(x)

    # second
    x = data_augmentation(x)
    x = preprocess_input(x)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280, activation='relu')(x)
    x = Dropout(0.5)(x)

    pooled_output = GlobalAveragePooling2D()(x)
    outputs = Dense(embedding_dim)(pooled_output)
    model = Model(inputs, outputs)

    return model


