import numpy as np
import pandas as pd
import helper
import tensorflow as tf

# image paths
df = pd.read_csv('/home/sarah/Deep-Learning/MS4/preprocessing/promis.csv')
image_paths = df['path'].to_numpy()
image_names = df['name'].to_numpy()

# make pairs
helper.create_pairs(image_paths, image_names)

pair_df = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/pairs.csv')
pair_1 = pair_df['image1'].to_numpy()
pair_2 = pair_df['image2'].to_numpy()
labels = pair_df['PairLabels'].to_numpy()

dataset = tf.data.Dataset.from_tensor_slices(((pair_1, pair_2), labels))
print('hier')

dataset = dataset.map(lambda pair, label: helper.load_images(pair[0], pair[1], label))

for element in dataset.take(5):
    print(element)
