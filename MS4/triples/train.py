import helper
import pandas as pd
import tensorflow as tf

##
BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 50
IMG_SHAPE = IMG_SIZE + (3,)

#-- GET DATA -----------------------------------------------------------------------------------------------------------
df_positive = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/triple_positive.csv', index_col=0)
df_negative = pd.read_csv('/home/sarah/Deep-Learning/MS4/data/triple_negative.csv', index_col=0)

df_positive['negative'] = df_negative['negative']

# shuffle dataframe
train_df = df_positive.sample(frac=1)
dataset_size = len(train_df)

#-- MAKE DATASET -------------------------------------------------------------------------------------------------------
anchor_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['anchor'].values)
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

# Zip datasets to form triplets
triplet_dataset = tf.data.Dataset.zip((anchor_dataset, positive_dataset))
dataset = tf.data.Dataset.zip((triplet_dataset, negative_dataset))

##
# -- SPLIT DATASET INTO TRAIN, VAL AND TEST ----------------------------------------------------------------------------
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
