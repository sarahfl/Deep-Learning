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
    img = tf.image.resize(img, [200, 200])
    return img, {'age_output': label_age, 'gender_output': label_gender, 'face_output': label_face}


##
BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 50
IMG_SHAPE = IMG_SIZE + (3,)

##
# -- GET DATA ----------------------------------------------------------------------------------------------------------
df_face = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/UTKFace.csv', index_col=0)
df_noFace = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/noFace.csv', index_col=0)

df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

print(df.head())
# shuffle dataframe
train_df = df.sample(frac=1)

##
# -- ONE HOT ENCODING --------------------------------------------------------------------------------------------------
one_hot_age = pd.get_dummies(train_df['age']).astype(int)
one_hot_gender = pd.get_dummies(train_df['gender']).astype(int)
one_hot_face = pd.get_dummies(train_df['face']).astype(int)

# convert to numpy-array
one_hot_age = one_hot_age.to_numpy()
one_hot_gender = one_hot_gender.to_numpy()
one_hot_face = one_hot_face.to_numpy()

dataset_size = len(df)
print('Größe des Datensets: ', dataset_size)

##
# -- MAKE DATASET ------------------------------------------------------------------------------------------------------
dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['path'].values, one_hot_age, one_hot_gender, one_hot_face))

dataset = dataset.map(load_and_preprocess_image)
print(dataset.element_spec)


##
# -- SPLIT DATASET INTO TRAIN, VAL AND TEST ----------------------------------------------------------------------------
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)

##
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

##
# -- BASE MODEL MOBILENETV2 --------------------------------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.summary()
print(len(base_model.trainable_variables))

base_model.trainable = False

##
# -- CUSTOM TOP LAYER --------------------------------------------------------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
inputs = tf.keras.Input(shape=(200, 200, 3))
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
output_face = tf.keras.layers.Dense(1, name='face_output')(x)

# COMBINE
model = tf.keras.Model(inputs, [output_age, output_gender, output_face])

##
# -- COMPILE THE MODEL -------------------------------------------------------------------------------------------------
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'age_output': tf.keras.losses.CategoricalCrossentropy(),
                    'gender_output': tf.keras.losses.CategoricalCrossentropy(),
                    'face_output': tf.keras.losses.BinaryCrossentropy()},
              metrics={'age_output': 'accuracy',
                       'gender_output': 'accuracy',
                       'face_output': 'accuracy'})

model.summary()
print(len(model.trainable_variables))

# save model summary to file
with open('model_summary_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(base_learning_rate, EPOCHS, BATCH_SIZE), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

##
# -- TRAIN THE MODEL ---------------------------------------------------------------------------------------------------
# Definiere den Early Stopping Callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=validation_dataset)

##
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# save the model
model.save('/home/sarah/Deep-Learning/MS3/model.keras')
