import configuration
import tensorflow as tf
import pandas as pd
import MS4.utils
import helper


pair_df = pd.read_csv(configuration.PAIR_PATH)
pair_1 = pair_df['image1'].to_numpy()
pair_2 = pair_df['image2'].to_numpy()
labels = pair_df['PairLabels'].to_numpy()

dataset = tf.data.Dataset.from_tensor_slices(((pair_1, pair_2), labels))
dataset = dataset.map(lambda pair, label: helper.load_images(pair[0], pair[1], label))

dataset_size = len(pair_df)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(configuration.BATCH_SIZE)
val_dataset = val_dataset.batch(configuration.BATCH_SIZE)
test_dataset = test_dataset.batch(configuration.BATCH_SIZE)

custom_objects = {"MS4.utils": MS4.utils, "euclidean_distance": MS4.utils.euclidean_distance}
model = tf.keras.models.load_model(configuration.MODEL_PATH, custom_objects=custom_objects)

predictions = model.predict(dataset)
print(predictions)

