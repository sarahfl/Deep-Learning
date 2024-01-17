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
    df_name_astype_category = df['name'].astype('category')
    code_to_category = dict(enumerate(df_name_astype_category.cat.categories))
    num_identities = len(df_name_astype_category.cat.categories)
    df["identity"] = df_name_astype_category.cat.codes
    df.sort_values(by='identity').to_csv('/tmp/promis_identity.csv', index=True)
    df.to_csv('/tmp/promis_identity_us.csv')
    df["image"] = df["path"].apply(utils.load_image)

    # pair is a list of image pairs
    # label is whether they are showing the same person or not

    pair, label = utils.make_pairs(images=df['image'], identities=df['identity'], num_identities=num_identities)

    # EXPLAIN PAIRS
    # pair_expl, label_expl = utils.make_pairs(images=df['path'], identities=df['identity'],
    #                                          num_identities=num_identities)
    #
    # df_explain = pd.DataFrame(
    #     {'path1': pair_expl[:, 0], 'path2': pair_expl[:, 1], 'label': label_expl.flatten()})
    # df_explain.to_csv('/tmp/promis_pair_expl.csv', index=True)


    # pair = pair[:10]
    # label = label[:10]
    #
    # # Assuming each element of pair is a tuple (image1, image2)
    # image1_list = [p[0] for p in pair]
    # image2_list = [p[1] for p in pair]
    #
    # # Assuming image1_list and image2_list are now lists of 3D NumPy arrays
    # dataset = Dataset.from_tensor_slices(([image1_list, image2_list], label))
    # print(dataset)
    #
    # print('Aufbau des Datensets: ', dataset.element_spec)

    dataset_size = len(df)
    ##
    # -- SPLIT DATASET INTO TRAIN, VAL AND TEST --
    # train=0.8, validation=0.1, test=0.1
    train_size = int(0.8 * dataset_size)
    val_size = int(0.1 * dataset_size)
    test_size = dataset_size - train_size - val_size

    # train_dataset = dataset.take(train_size)
    # val_dataset = dataset.skip(train_size).take(val_size)
    # test_dataset = dataset.skip(train_size + val_size)
    #
    # train_dataset = train_dataset.batch(configuration.BATCH_SIZE)
    # val_dataset = val_dataset.batch(configuration.BATCH_SIZE)
    # test_dataset = test_dataset.batch(configuration.BATCH_SIZE)
    #
    # train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
    # test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    dataset_size = len(label)
    train_size = int(0.7 * dataset_size)
    val_size = int(0.15 * dataset_size)
    test_size = dataset_size - train_size - val_size

    pair_train = pair[:train_size]
    pair_val = pair[train_size:train_size + val_size]
    pair_test = pair[train_size + val_size:]

    label_train = label[:train_size]
    label_val = label[train_size:train_size + val_size]
    label_test = label[train_size + val_size:]

    return (pair_train, label_train), (pair_val, label_val), (pair_test, label_test)
    # print("Created datasets")
    # return train_dataset, val_dataset, test_dataset


(pair_train, label_train), (pair_val, label_val), (pair_test, label_test) = do_promis()
# train_dataset, val_dataset, test_dataset = do_promis()

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
model.compile(loss="binary_crossentropy", optimizer="adam",
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
    [pair_train[:, 0], pair_train[:, 1]], label_train[:],
    validation_data=([pair_val[:, 0], pair_val[:, 1]], label_val[:]),
    # train_dataset,
    # validation_data=val_dataset,
    batch_size=configuration.BATCH_SIZE,
    epochs=configuration.EPOCHS,
    callbacks=[early_stopping])

model.save(configuration.MODEL_PATH)
logging.info("Plotting training history...")
utils.plot_training(history.history, configuration.PLOT_PATH)
