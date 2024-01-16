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
from tensorflow.data import AUTOTUNE
from tensorflow.data import Dataset
from tensorflow.keras.datasets import mnist
from tensorflow.data import Dataset
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


def do_mnist():
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


def do_promis():
    logging.info("[INFO] loading promis dataset...")
    df = utils.read_csv()
    df["image"] = df["path"].apply(utils.load_and_preprocess_image)
    # Check encoding
    # print(df['name'][0])
    code_to_category = dict(enumerate(df['name'].astype('category').cat.categories))
    df['name'] = df['name'].astype('category').cat.codes
    # print(df['name'][0])
    # print(df['name'].map(code_to_category)[0])

    # pair is a list of image pairs
    # label is whether they are showing the same person or not
    pair, label = utils.make_pairs(df['image'], df['name'])
    print("Before slicing")
    dataset = Dataset.from_tensor_slices(([pair[:, 0], pair[:, 1]], label))
    print("After slicing")
    print('Aufbau des Datensets: ', dataset.element_spec)
    dataset_size = len(df)
    ##
    # -- SPLIT DATASET INTO TRAIN, VAL AND TEST --
    # train=0.8, validation=0.1, test=0.1
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size).take(val_size)
    test_dataset = dataset.skip(train_size + val_size)

    train_dataset = train_dataset.batch(configuration.BATCH_SIZE)
    val_dataset = val_dataset.batch(configuration.BATCH_SIZE)
    test_dataset = test_dataset.batch(configuration.BATCH_SIZE)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    # dataset_size = len(label)
    # train_size = int(0.7 * dataset_size)
    # val_size = int(0.15 * dataset_size)
    # test_size = dataset_size - train_size - val_size
    #
    # pair_train = pair[:train_size]
    # pair_val = pair[train_size:train_size+val_size]
    # pair_test = pair[train_size + val_size:]
    #
    # label_train = label[:train_size]
    # label_val = label[train_size:train_size+val_size]
    # label_test = label[train_size + val_size:]

    # return (pair_train, label_train), (pair_val, label_val), (pair_test, label_test)
    print("Created datasets")
    return train_dataset, val_dataset, test_dataset


# (pair_train, label_train), (pair_val, label_val), (pair_test, val_test) = do_promis()
train_dataset, val_dataset, test_dataset = do_promis()

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
model.summary()
# callbacks
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitors the validation loss
    patience=3,  # Number of epochs with no improvement after which training will stop
    restore_best_weights=True  # Restores the best model weights based on the monitored quantity
)

# train the model
logging.info("Training model...")
history = model.fit(
    # [pair_train[:, 0], pair_train[:, 1]], label_train[:],
    # validation_data=([pair_val[:, 0], pair_val[:, 1]], label_val[:]),
    train_dataset,
    val_dataset,
    batch_size=configuration.BATCH_SIZE,
    epochs=configuration.EPOCHS,
    callbacks=[early_stopping])

model.save(configuration.MODEL_PATH)
logging.info("Plotting training history...")
utils.plot_training(history.history, configuration.PLOT_PATH)
