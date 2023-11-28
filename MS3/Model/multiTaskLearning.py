##
from random import shuffle

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import numpy as np


def load_and_preprocess_image(image_path, label_age, label_gender, label_face):
    # Lade das Bild
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return img, {'age_output': label_age, 'gender_output': label_gender, 'face_output': label_face}



##

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 50
IMG_SHAPE = IMG_SIZE + (3,)

##
# -- GET DATA ----------------------------------------------------------------------------------------------------------
df_face = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/UTKFace.csv')
df_noFace = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/noFace.csv')

df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

# shuffle dataframe
train_df = df.sample(frac=1)

##
# -- MAKE DATASET ------------------------------------------------------------------------------------------------------
#TODO: one hot encoding
dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['path'].values, train_df['age'].values, train_df['gender'].values, train_df['face']))

dataset = dataset.map(load_and_preprocess_image)

train_size = int(0.8 * len(df))
train_dataset = dataset.take(train_size)
test_dataset = dataset.skip(train_size)

##
##
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# Laden des vorab trainierten MobileNetV2-Modells
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)

base_model.summary()
print(len(base_model.trainable_variables))

base_model.trainable = False
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = preprocess_input(inputs)

x = base_model(x, training=False)

x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1280, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# OUTPUT AGE
output_age = tf.keras.layers.Dense(8, activation='softmax', name='age_output')(x)

# OUTPUT GENDER
output_gender = tf.keras.layers.Dense(3, activation='softmax', name='gender_output')(x)

# OUTPUT FACE
output_face = tf.keras.layers.Dense(1, activation='sigmoid', name='face_output')()

# COMBINE
model = tf.keras.Model(inputs, [output_age, output_gender, output_face])

base_learning_rate = 0.0001
# Kompilieren des Modells
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'age_output': tf.keras.losses.CategoricalCrossentropy(),
                    'gender_output': tf.keras.losses.CategoricalCrossentropy(),
                    'face_output': tf.keras.losses.BinaryCrossentropy()},
              metrics={'age_output': 'accuracy',
                       'gender_output': 'accuracy',
                       'face_output': 'accuracy'})

# Modellzusammenfassung anzeigen
model.summary()
