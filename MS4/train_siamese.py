"""
https://pyimagesearch.com/2020/11/30/siamese-networks-with-keras-tensorflow-and-deep-learning/
"""

from MS4.siamese import build_siamese_model
from MS4 import configuration
from MS4 import utils
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Lambda
from tensorflow.keras.callbacks import EarlyStopping
# from tensorflow.data import AUTOTUNE
# from tensorflow.data import Dataset
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import plot_model
from tensorflow.keras.optimizers import Adam
import numpy as np
import logging
import os
import pandas as pd

os.makedirs(os.path.dirname(configuration.LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(configuration.LOG_PATH),
        logging.StreamHandler()
    ]
)

import helper
import tensorflow as tf

# image paths
df = pd.read_csv('MS4/preprocessing/promis.csv')
image_paths = df['path'].to_numpy()
image_names = df['name'].to_numpy()

# make pairs
if not os.path.isfile('MS4/data/pairs.csv'):
    logging.info("Creating pairs...")
    helper.create_pairs(image_paths, image_names)
else:
    logging.info("Pairs found. Continuing...")

pair_df = pd.read_csv('MS4/data/pairs.csv')
pair_1 = pair_df['image1'].to_numpy()
pair_2 = pair_df['image2'].to_numpy()
labels = pair_df['PairLabels'].to_numpy()

dataset = tf.data.Dataset.from_tensor_slices(((pair_1, pair_2), labels))
print('hier')

dataset = dataset.map(lambda pair, label: helper.load_images(pair[0], pair[1], label))

for element in dataset.take(5):
    print(element)


print('Aufbau des Datensets: ', dataset.element_spec)

##
# -- SPLIT DATASET INTO TRAIN, VAL AND TEST ----------------------------------------------------------------------------
# train=0.8, validation=0.1, test=0.1
dataset_size = dataset_size = len(df)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(configuration.BATCH_SIZE)
val_dataset = val_dataset.batch(configuration.BATCH_SIZE)
test_dataset = test_dataset.batch(configuration.BATCH_SIZE)

##
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# configure siamese
logging.info("Building siamese network...")

# Our model takes two inputs
imgA = Input(shape=configuration.IMG_SHAPE)
imgB = Input(shape=configuration.IMG_SHAPE)
# takes a common core
featureExtractor = build_siamese_model(configuration.IMG_SHAPE)
# one instance for both paths for shared weights
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)
# calculates the Euclidean distance between those two
distance = Lambda(utils.euclidean_distance, name='lambda_eucl_dist')([featsA, featsB])
# and returns in [0,1] depending on whether it is the same person or not.
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
logging.info("Compiling model...")
model.compile(loss="binary_crossentropy", optimizer=Adam(lr=0.001),
              metrics=["accuracy"])
model.summary()
output_path = '/tmp/model_1.png'
plot_model(model, to_file=output_path, show_shapes=True, show_layer_names=True)

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors the validation loss
    patience=10,  # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores the best model weights based on the monitored quantity
)

# train the model
logging.info("Training model...")
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    # train_dataset,
    # validation_data=val_dataset,
    batch_size=configuration.BATCH_SIZE,
    epochs=configuration.EPOCHS,
    callbacks=[early_stopping])

training_history = history.history
history_df = pd.DataFrame(training_history)
history_df.to_csv(configuration.TRAINING_HISTORY_PATH, index=False)


model.save(configuration.MODEL_PATH)
logging.info("Plotting training history...")
utils.plot_training(history.history, configuration.PLOT_PATH)
