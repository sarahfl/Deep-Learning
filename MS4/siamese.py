from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import RandomFlip
from tensorflow.keras.layers import RandomRotation, Conv2D, MaxPooling2D
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Sequential
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def build_siamese_model(input_shape, embedding_dim=48):
    # specify the inputs for the feature extractor network
    inputs = Input(shape=input_shape)
    base_model = MobileNetV2(
        weights='imagenet',  # Load weights pre-trained on ImageNet.
        input_shape=input_shape,  # Ensure input has three channels (RGB)
        include_top=False,
    )

    x = preprocess_input(inputs)
    x = base_model(x)
    x = GlobalAveragePooling2D()(x)
    x = Dense(1280, activation='relu')(x)
    x = Dropout(0.5)(x)

    # second
    # x = base_model(x)
    # x = GlobalAveragePooling2D()(x)
    # x = Dense(1280, activation='relu')(x)
    # x = Dropout(0.5)(x)
    #
    # # define the first set of CONV => RELU => POOL => DROPOUT layers
    # x = Conv2D(64, (3, 3), padding="same", activation="relu")(inputs)  # Adjust filter size to 3x3
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.3)(x)
    #
    # # second set of CONV => RELU => POOL => DROPOUT layers
    # x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)  # Adjust filter size to 3x3
    # x = MaxPooling2D(pool_size=(2, 2))(x)
    # x = Dropout(0.3)(x)

    # pooled_output = GlobalAveragePooling2D()(x)
    outputs = Dense(embedding_dim)(x)
    model = Model(inputs, outputs, name="siam_core")

    return model


