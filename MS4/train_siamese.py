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
from tensorflow.keras.datasets import mnist
import numpy as np
import logging
import os

os.makedirs(os.path.dirname(configuration.LOG_PATH), exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(configuration.LOG_PATH),
        logging.StreamHandler()
    ]
)

logging.info("[INFO] loading MNIST dataset...")
(trainX, trainY), (testX, testY) = mnist.load_data()  # ~/.keras/datasets/mnist.npz

# utils.display_image(trainX, trainY, 7)

trainX = trainX / 255.0  # scale [0-1]
testX = testX / 255.0  # scale [0-1]
# add a channel dimension to the images
trainX = np.expand_dims(trainX, axis=-1)
testX = np.expand_dims(testX, axis=-1)
# prepare the positive and negative pairs
logging.info("Preparing positive and negative pairs...")
(pairTrain, labelTrain) = utils.make_pairs(trainX, trainY)
(pairTest, labelTest) = utils.make_pairs(testX, testY)

# configure siamese
logging.info("Building siamese network...")
imgA = Input(shape=configuration.IMG_SHAPE)
imgB = Input(shape=configuration.IMG_SHAPE)

# one instance for shared weights
featureExtractor = build_siamese_model(configuration.IMG_SHAPE)
featsA = featureExtractor(imgA)
featsB = featureExtractor(imgB)

distance = Lambda(utils.euclidean_distance)([featsA, featsB])
outputs = Dense(1, activation="sigmoid")(distance)
model = Model(inputs=[imgA, imgB], outputs=outputs)

# compile the model
logging.info("Compiling model...")
model.compile(loss="binary_crossentropy", optimizer="adam",
              metrics=["accuracy"])

# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors the validation loss
    patience=3,  # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores the best model weights based on the monitored quantity
)

# train the model
logging.info("Training model...")
history = model.fit(
    [pairTrain[:, 0], pairTrain[:, 1]], labelTrain[:],
    validation_data=([pairTest[:, 0], pairTest[:, 1]], labelTest[:]),
    batch_size=configuration.BATCH_SIZE,
    epochs=configuration.EPOCHS,
    callbacks=[early_stopping])

model.save(configuration.MODEL_PATH)
logging.info("Plotting training history...")
utils.plot_training(history.history, configuration.PLOT_PATH)

