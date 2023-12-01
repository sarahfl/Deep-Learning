##
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import numpy as np

modelType = 'model1_classification'


def load_and_preprocess_image(image_path, label_age, label_gender, label_face):
    """
    Apply tensor transformation on image.
    :param image_path: absolut path to file
    :param label_age: output channel age
    :param label_gender: output channel gender
    :param label_face: output channel face
    :return: Image as 3D tensor and dictionary of labels which includes the mapping of labels to output channels
    """
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)  # image to tensor
    img = tf.image.resize(img, [200, 200])  # define image size
    return img, {'age_output': label_age, 'gender_output': label_gender, 'face_output': label_face}


##
BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 20
IMG_SHAPE = IMG_SIZE + (3,)

##
# -- GET DATA ----------------------------------------------------------------------------------------------------------
df_face = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/Face_classification.csv', index_col=0)
df_noFace = pd.read_csv('/home/sarah/Deep-Learning/MS3/preprocessing/Data/noFace_classification.csv', index_col=0)
df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

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
print('Aufbau des Datensets: ', dataset.element_spec)

##
# -- SPLIT DATASET INTO TRAIN, VAL AND TEST ----------------------------------------------------------------------------
# train=0.8, validation=0.1, test=0.1
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=True)

##
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

##
# -- DATA AUGMENTATION -------------------------------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2),
])

##
# -- BASE MODEL MOBILENETV2 --------------------------------------------------------------------------------------------
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=IMG_SHAPE)
base_model.summary()
print('Anzahl der trainierbaren Variablen: ', len(base_model.trainable_variables))

base_model.trainable = False

##
# -- CUSTOM TOP LAYER --------------------------------------------------------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(200, 200, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1280, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

# OUTPUT AGE
output_age = tf.keras.layers.Dense(8, activation='softmax', name='age_output')(x)

# OUTPUT GENDER
output_gender = tf.keras.layers.Dense(3, activation='softmax', name='gender_output')(x)

# OUTPUT FACE
output_face = tf.keras.layers.Dense(2, activation='softmax', name='face_output')(x)

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
print('Anzahl der trainierbaren Variablen: ', len(model.trainable_variables))

# save model summary to file
with open('{}/model_summary_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(modelType, base_learning_rate, EPOCHS, BATCH_SIZE),
          'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

##
# -- TRAIN THE MODEL ---------------------------------------------------------------------------------------------------
# Early Stopping Callback
# early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=validation_dataset)

##
# -- SAVE HISTORY AND MODEL --------------------------------------------------------------------------------------------
# save Training-History for each output channel
history_age = history.history['age_output_accuracy']
val_history_age = history.history['val_age_output_accuracy']
loss_age = history.history['age_output_loss']
val_loss_age = history.history['val_age_output_loss']

history_gender = history.history['gender_output_accuracy']
val_history_gender = history.history['val_gender_output_accuracy']
loss_gender = history.history['gender_output_loss']
val_loss_gender = history.history['val_gender_output_loss']

history_face = history.history['face_output_accuracy']
val_history_face = history.history['val_face_output_accuracy']
loss_face = history.history['face_output_loss']
val_loss_face = history.history['val_face_output_loss']

# make Dataframe for each output channel
hist_df_age = pd.DataFrame({
    'accuracy': history_age,
    'val_accuracy': val_history_age,
    'loss': loss_age,
    'val_loss': val_loss_age
})

hist_df_gender = pd.DataFrame({
    'accuracy': history_gender,
    'val_accuracy': val_history_gender,
    'loss': loss_gender,
    'val_loss': val_loss_gender
})

hist_df_face = pd.DataFrame({
    'accuracy': history_face,
    'val_accuracy': val_history_face,
    'loss': loss_face,
    'val_loss': val_loss_face
})

# save each Dataframe to csv
hist_csv_file_age = '{}/history_age.csv'.format(modelType)
with open(hist_csv_file_age, mode='w') as f_age:
    hist_df_age.to_csv(f_age)

hist_csv_file_gender = '{}/history_gender.csv'.format(modelType)
with open(hist_csv_file_gender, mode='w') as f_gender:
    hist_df_gender.to_csv(f_gender)

hist_csv_file_face = '{}/history_face.csv'.format(modelType)
with open(hist_csv_file_face, mode='w') as f_face:
    hist_df_face.to_csv(f_face)

# save whole model
model.save('/home/sarah/Deep-Learning/MS3/Model/{}/model.keras'.format(modelType))

##
# -- EVALUATE VALIDATION AND TEST DATA ---------------------------------------------------------------------------------
eval_test = model.evaluate(test_dataset)
eval_val = model.evaluate(validation_dataset)

# Losses and Accuracies Test
losses_test = eval_test[:3]
accuracies_test = eval_test[3:]

# Losses and Accuracies Validation
losses_val = eval_val[:3]
accuracies_val = eval_val[3:]

df_test = pd.DataFrame({
    'Loss Age': [losses_test[0]],
    'Loss Gender': [losses_test[1]],
    'Loss Face': [losses_test[2]],
    'Accuracy Age': [accuracies_test[0]],
    'Accuracy Gender': [accuracies_test[1]],
    'Accuracy Face': [accuracies_test[2]]
})

df_val = pd.DataFrame({
    'Loss Age': [losses_val[0]],
    'Loss Gender': [losses_val[1]],
    'Loss Face': [losses_val[2]],
    'Accuracy Age': [accuracies_val[0]],
    'Accuracy Gender': [accuracies_val[1]],
    'Accuracy Face': [accuracies_val[2]]
})

print('Evalutation Validation ----------------------------')
print(df_val)
print('Evalutation Test ----------------------------------')
print(df_test)

# save to csv
csv_file_test = '{}/test_metrics.csv'.format(modelType)
csv_file_val = '{}/validation_metrics.csv'.format(modelType)
df_test.to_csv(csv_file_test, index=False)
df_val.to_csv(csv_file_val, index=False)
