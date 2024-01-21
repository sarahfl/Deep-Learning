import helper
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 10
IMG_SHAPE = IMG_SIZE + (3,)
name = 'model7'

# -- GET DATA ----------------------------------------------------------------------------------------------------------
df_positive = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/triple_positive.csv', index_col=0)
df_negative = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/triple_negative.csv', index_col=0)
df_positive['negative'] = df_negative['negative']

# shuffle dataframe
train_df = df_positive.sample(frac=1)
dataset_size = len(train_df)

# -- MAKE DATASET ------------------------------------------------------------------------------------------------------
anchor_dataset = tf.data.Dataset.from_tensor_slices(
    train_df['anchor'].values
)
positive_dataset = tf.data.Dataset.from_tensor_slices(
    train_df['positive'].values
)
negative_dataset = tf.data.Dataset.from_tensor_slices(
    train_df['negative'].values
)
anchor_dataset = anchor_dataset.map(helper.load_and_preprocess_image)
positive_dataset = positive_dataset.map(helper.load_and_preprocess_image)
negative_dataset = negative_dataset.map(helper.load_and_preprocess_image)

# Zip datasets to form triples
dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset, negative_dataset))

# -- PLOT EXAMPLES FROM DATASET ----------------------------------------------------------------------------------------
sample_dataset = dataset.take(3)
for anchor_img, positive_img, negative_img in sample_dataset:
    anchor_img = anchor_img.numpy()
    positive_img = positive_img.numpy()
    negative_img = negative_img.numpy()

    plt.figure(figsize=(9, 3))

    plt.subplot(1, 3, 1)
    plt.imshow(anchor_img)
    plt.title('Anchor')

    plt.subplot(1, 3, 2)
    plt.imshow(positive_img)
    plt.title('Positive')

    plt.subplot(1, 3, 3)
    plt.imshow(negative_img)
    plt.title('Negative')

    plt.show()

# -- SPLIT DATASET INTO TRAIN, VALIDATION AND TEST ---------------------------------------------------------------------
# train=0.8, validation=0.1, test=0.1
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

# -- PREFETCH ----------------------------------------------------------------------------------------------------------
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

print('Aufbau des Datensets: ', dataset.element_spec)

# -- MODEL -------------------------------------------------------------------------------------------------------------
# get model for feature extraction
embedding = helper.get_model(IMG_SHAPE)

# INPUT LAYERS (anchor, positive, negative)
anchor_input = layers.Input(shape=IMG_SHAPE, name="Anchor_Input")
positive_input = layers.Input(shape=IMG_SHAPE, name="Positive_Input")
negative_input = layers.Input(shape=IMG_SHAPE, name="Negative_Input")

# PREPROCESSING
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

# COMPUTE DISTANCE
distance_layer = helper.DistanceLayer2()
anchor_embedding = embedding(preprocess_input(tf.cast(anchor_input, tf.float32)))
positive_embedding = embedding(preprocess_input(tf.cast(positive_input, tf.float32)))
negative_embedding = embedding(preprocess_input(tf.cast(negative_input, tf.float32)))

distance = distance_layer([anchor_embedding, positive_embedding, negative_embedding])

# -- CREATE MODEL ------------------------------------------------------------------------------------------------------
siamese_network = tf.keras.Model(
    inputs=[anchor_input, positive_input, negative_input],
    outputs=distance,
    name="Siamese_Network")

# show network structure
siamese_network.summary()
tf.keras.utils.plot_model(siamese_network, show_shapes=True, show_layer_names=True)

# -- TRAIN MODEL -------------------------------------------------------------------------------------------------------
siamese_model = helper.SiameseModel(siamese_network)
siamese_model.compile(optimizer=tf.keras.optimizers.Adam(0.001))
history = siamese_model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

history_df = pd.DataFrame(history.history)
history_df.to_csv('/home/sarah/Deep-Learning/MS4/triples/Model/{}/history.csv'.format(name), index=False)

siamese_model.save_weights("/home/sarah/Deep-Learning/MS4/triples/Model/{}/final_weights.h5".format(name))

