import helper
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 10
IMG_SHAPE = IMG_SIZE + (3,)
name = 'model6'

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
distance_layer = helper.DistanceLayer()
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

# build custom siamese keras model
siamese_model = helper.SiameseModel(siamese_network)

# compile model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
siamese_model.compile(optimizer=optimizer, weighted_metrics=[])

# -- TRAIN MODEL -------------------------------------------------------------------------------------------------------
loss, metrics = helper.train_model(siamese_model, train_dataset, EPOCHS, val_dataset, BATCH_SIZE)

# save weights
siamese_model.save_weights("/home/sarah/Deep-Learning/MS4/triples/Model/{}/final_weights.h5".format(name))

# -- EVALUATE ON TEST DATA ---------------------------------------------------------------------------------------------
evaluation_metrics = helper.evaluate_model(siamese_model, test_dataset)
result_filename = '/home/sarah/Deep-Learning/MS4/triples/Model/{}/result_test_loss.txt'.format(name)
with open(result_filename, 'w') as file:
    print("Test Loss:", evaluation_metrics, file=file)

# plot metrics
helper.plot_metrics(loss, metrics, name)

# save metrics to csv
csv_filename = "/home/sarah/Deep-Learning/MS4/triples/Model/{}/metrics.csv".format(name)
helper.save_metrics_to_csv(loss, metrics, csv_filename)

# -- TEST AND ANALYZE --------------------------------------------------------------------------------------------------
encode_model = helper.get_model(IMG_SHAPE)
filepath = "/home/sarah/Deep-Learning/MS4/triples/Model/{}/final_weights.h5".format(name)
encode_model.load_weights(filepath, by_name=True)

# Analyze Test Dataset
similarities = helper.calculate_similarity_tensor(embedding, test_dataset)
tupels = helper.convert_to_tuples_tensor(similarities)
differences = helper.calculate_differences_within_tuples(tupels)
avg = helper.average(differences)
print(avg)

# Doppelganger Dataset
similarities_doppelganger = helper.calculate_similarity(embedding, helper.image_pairs)
tupels_doppelganger = helper.convert_to_tuples(similarities_doppelganger)
differences_doppelganger = helper.calculate_differences_within_tuples(tupels_doppelganger)
avg_doppelganger = helper.average(differences_doppelganger)
print(avg_doppelganger)
helper.plot_image_triples(helper.image_pairs, similarities_doppelganger, name)

avg_filename = '/home/sarah/Deep-Learning/MS4/triples/Model/{}/avg.txt'.format(name)
with open(avg_filename, 'w') as file:
    print("Test Avg:", avg, file=file)
    print("Doppelganger Avg:", avg_doppelganger, file=file)
    print('Differences:', differences_doppelganger, file=file)

