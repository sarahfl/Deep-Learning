##
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import numpy as np
import keras
import os
import cv2
from scipy import interpolate
from sklearn.metrics import mean_absolute_error

print(os.path.dirname(__file__))
model_type = 'model7_regression'

if not os.path.exists(f"MS3/Model/{model_type}"):
    os.mkdir(f"MS3/Model/{model_type}")


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
EPOCHS = 50
IMG_SHAPE = IMG_SIZE + (3,)

##
# -- GET DATA ----------------------------------------------------------------------------------------------------------
df_face = pd.read_csv('MS3/Model/data/Face_regression.csv', index_col=0)
df_noFace = pd.read_csv('MS3/Model/data/noFace_regression.csv', index_col=0)
# delta = len(df_noFace) - 12500
# drop_indices = np.random.choice(df_noFace.index, delta, replace=False)
# df_subset_noFace = df_noFace.drop(drop_indices)

df = pd.concat([df_face, df_noFace], axis=0, ignore_index=True)

# shuffle dataframe
train_df = df.sample(frac=1)

##
# -- ONE HOT ENCODING --------------------------------------------------------------------------------------------------
age_labels = train_df['age'].values.astype(int)
one_hot_gender = pd.get_dummies(train_df['gender']).astype(int)
one_hot_face = pd.get_dummies(train_df['face']).astype(int)

# Begin of Normalize
# age_mean = age_labels.mean()
# age_std = age_labels.std()
# age_labels = (age_labels - age_mean) / age_std  # Normalize using mean and standard deviation
# End of Normalize

# convert to numpy-array
one_hot_gender = one_hot_gender.to_numpy()
one_hot_face = one_hot_face.to_numpy()

dataset_size = len(df)
print('Größe des Datensets: ', dataset_size)

##
# -- MAKE DATASET ------------------------------------------------------------------------------------------------------
dataset = tf.data.Dataset.from_tensor_slices(
    (train_df['path'].values, age_labels, one_hot_gender, one_hot_face))

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

train_dataset = train_dataset.batch(BATCH_SIZE)
val_dataset = val_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.batch(BATCH_SIZE)

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

# -- FINE TUNING -------------------------------------------------------------------------------------------------------
print("Number of layers in the base model: ", len(base_model.layers))
fine_tune_at = 130
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

##
# -- CUSTOM TOP LAYER --------------------------------------------------------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=(200, 200, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1280, activation='relu')(x)
x = tf.keras.layers.Dropout(0.5)(x)

x_age = tf.keras.layers.Dense(640, activation='relu')(x)
# x_age = tf.keras.layers.Dropout(0.5)(x_age)
x_age = tf.keras.layers.Dense(320, activation='relu')(x_age)
# x_age = tf.keras.layers.Dropout(0.5)(x_age)
x_age = tf.keras.layers.Dense(160, activation='relu')(x_age)
# x_age = tf.keras.layers.Dropout(0.5)(x_age)
x_age = tf.keras.layers.Dense(80, activation='relu')(x_age)
x_age = tf.keras.layers.Dropout(0.5)(x_age)

# OUTPUT AGE
output_age = tf.keras.layers.Dense(1, activation='linear', name='age_output')(x_age) # x

# OUTPUT GENDER
output_gender = tf.keras.layers.Dense(3, activation='softmax', name='gender_output')(x)

# OUTPUT FACE
output_face = tf.keras.layers.Dense(2, activation='softmax', name='face_output')(x)

# COMBINE
model = tf.keras.Model(inputs, [output_age, output_gender, output_face])


##
@tf.keras.utils.register_keras_serializable(package='Custom', name='custom_mse')
class CustomMSE(tf.keras.losses.Loss):
    def __init__(self, name='custom_mse'):
        super().__init__(name=name)

    def call(self, y_true, y_pred):
        loss = tf.where(tf.math.not_equal(y_true, 0),
                        tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - y_pred)),
                        0.0)  # Set loss to 0 where y_true is 0
        return loss

    def get_config(self):
        return {'name': self.name}


##
# -- COMPILE THE MODEL -------------------------------------------------------------------------------------------------

# def custom_mse(y_true, y_pred):
#    # Calculate the mean squared error only where y_true is non-zero
#    loss = tf.where(tf.math.not_equal(y_true, 0),
#                    # cast one to float or the other to int
#                    tf.reduce_mean(tf.square(tf.cast(y_true, tf.float32) - y_pred)),
#                    0.0)  # Set loss to 0 where y_true is 0
#    return loss


base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'age_output': CustomMSE(),
                    'gender_output': tf.keras.losses.CategoricalCrossentropy(),
                    'face_output': tf.keras.losses.BinaryCrossentropy()},
              metrics={'age_output': 'mae',
                       'gender_output': 'accuracy',
                       'face_output': 'accuracy'}
              )

model.summary()
print('Anzahl der trainierbaren Variablen: ', len(model.trainable_variables))

# save model summary to file
with open('MS3/Model/{}/model_summary_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(model_type, base_learning_rate, EPOCHS,
                                                                           BATCH_SIZE),
          'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

##
# -- TRAIN THE MODEL ---------------------------------------------------------------------------------------------------
# Early Stopping Callback
early_stopping_age = keras.callbacks.EarlyStopping(monitor='val_age_output_mae', patience=10, restore_best_weights=True)
# early_stopping_gender = keras.callbacks.EarlyStopping(monitor='val_gender_output_accuracy', patience=5, restore_best_weights=True)
# early_stopping_face = keras.callbacks.EarlyStopping(monitor='val_face_output_accuracy', patience=5, restore_best_weights=True)


history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=validation_dataset,
                    callbacks=[early_stopping_age])  # callbacks=[early_stopping_age]
##
age_loss = history.history['age_output_loss']
age_loss_val = history.history['val_age_output_loss']
gender_loss = history.history['gender_output_loss']
gender_loss_val = history.history['val_gender_output_loss']
face_loss = history.history['face_output_loss']
face_loss_val = history.history['val_face_output_loss']

epochs = range(1, len(age_loss) + 1)

# Plotting each loss separately
plt.figure(figsize=(10, 5))

plt.subplot(1, 3, 1)
plt.plot(epochs, age_loss, label='Age Loss')
plt.plot(epochs, age_loss_val, label='Age Loss Val')
plt.title('Age Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(epochs, gender_loss, label='Gender Loss')
plt.plot(epochs, gender_loss_val, label='Gender Loss Val')
plt.title('Gender Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(epochs, face_loss, label='Face Loss')
plt.plot(epochs, face_loss_val, label='Face Loss_val')
plt.title('Face Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(f"MS3/Model/{model_type}/plot_loss.png")
plt.show()
##
# -- SAVE HISTORY AND MODEL --------------------------------------------------------------------------------------------
# save Training-History for each output channel
history_age = history.history['age_output_mae']
val_history_age = history.history['val_age_output_mae']
loss_age = history.history['age_output_loss']
val_loss_age = history.history['val_age_output_loss']

history_gender = history.history['gender_output_accuracy']
val_history_gender = history.history['val_age_output_mae']
loss_gender = history.history['gender_output_loss']
val_loss_gender = history.history['val_gender_output_loss']

history_face = history.history['face_output_accuracy']
val_history_face = history.history['val_age_output_mae']
loss_face = history.history['face_output_loss']
val_loss_face = history.history['val_face_output_loss']

# make Dataframe for each output channel
hist_df_age = pd.DataFrame({
    'mae': history_age,
    'val_mae': val_history_age,
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
hist_csv_file_age = 'MS3/Model/{}/history_age.csv'.format(model_type)
with open(hist_csv_file_age, mode='w') as f_age:
    hist_df_age.to_csv(f_age)

hist_csv_file_gender = 'MS3/Model/{}/history_gender.csv'.format(model_type)
with open(hist_csv_file_gender, mode='w') as f_gender:
    hist_df_gender.to_csv(f_gender)

hist_csv_file_face = 'MS3/Model/{}/history_face.csv'.format(model_type)
with open(hist_csv_file_face, mode='w') as f_face:
    hist_df_face.to_csv(f_face)

# save whole model
model.save('MS3/Model/{}/model.keras'.format(model_type))

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
    'Mae Age': [accuracies_test[0]],
    'Accuracy Gender': [accuracies_test[1]],
    'Accuracy Face': [accuracies_test[2]]
})

df_val = pd.DataFrame({
    'Loss Age': [losses_val[0]],
    'Loss Gender': [losses_val[1]],
    'Loss Face': [losses_val[2]],
    'Mae Age': [accuracies_val[0]],
    'Accuracy Gender': [accuracies_val[1]],
    'Accuracy Face': [accuracies_val[2]]
})

print('Evalutation Validation ----------------------------')
print(df_val)
print('Evalutation Test ----------------------------------')
print(df_test)

# save to csv
csv_file_test = 'MS3/Model/{}/test_metrics.csv'.format(model_type)
csv_file_val = 'MS3/Model/{}/validation_metrics.csv'.format(model_type)
df_test.to_csv(csv_file_test, index=False)
df_val.to_csv(csv_file_val, index=False)

##
# predict test data
# tf.keras.utils.get_custom_objects()['CustomMSE'] = CustomMSE
# model = tf.keras.models.load_model('MS3/Model/{}/model.keras'.format(model_type))
predictions = model.predict(test_dataset)
##
# Predict single image for testing purposes
# x = cv2.imread('MS3/Model/data/UTKFace/21_0_face_9793.jpg')
# predictions = model.predict(np.expand_dims(x, axis=0))
##
# Extracting predictions

predicted_age = predictions[0].flatten()
predicted_gender = np.argmax(predictions[1], axis=1)
predicted_face = np.argmax(predictions[2], axis=1)

# Extracting true labels from the test dataset
true_labels_age = np.concatenate([label['age_output'] for _, label in test_dataset], axis=0)
true_labels_gender = np.argmax(np.concatenate([label['gender_output'] for _, label in test_dataset], axis=0), axis=1)
true_labels_face = np.argmax(np.concatenate([label['face_output'] for _, label in test_dataset], axis=0), axis=1)

# De-Normalize
# predicted_age = predicted_age * age_std + age_mean
# true_labels_age = true_labels_age * age_std + age_mean
# true_labels_age_original = np.round(true_labels_age).astype(int)
# End De-Normalize
predicted_age = np.round(predicted_age).astype(int)

remove_zeros = np.where(true_labels_age == 0)[0]

true_labels_age_nz = np.delete(true_labels_age, remove_zeros, axis=0)
predicted_age_nz = np.delete(predicted_age, remove_zeros, axis=0)

# Calculate MAE
mae = mean_absolute_error(true_labels_age_nz, predicted_age_nz)
print("MAE", mae)
# Calculate upper and lower boundaries for MAE
upper_bound_original = true_labels_age_nz + mae / 2
lower_bound_original = true_labels_age_nz - mae / 2
##
# Plotting predicted vs. true ages with MAE and ideal line
plt.figure(figsize=(8, 6))
plt.scatter(true_labels_age_nz, predicted_age_nz, alpha=0.3)
plt.plot(true_labels_age_nz, true_labels_age_nz, color='red', linestyle='-', label='Ideal')
plt.plot(true_labels_age_nz, upper_bound_original, color='green', linestyle='--',
         label=f'MAE +{mae / 2:.2f}')
plt.plot(true_labels_age_nz, lower_bound_original, color='green', linestyle='--',
         label=f'MAE -{mae / 2:.2f}')
plt.xlabel('True Age')
plt.ylabel('Predicted Age')
plt.title('Predicted vs. True Age')
plt.legend()
plt.savefig(f"MS3/Model/{model_type}/plot_upper_lower.png")
plt.show()
##
years = np.unique(predicted_age_nz)
years_mae = []
for year in years:
    predicted_for_year = predicted_age_nz[true_labels_age_nz == year]
    single_errors = predicted_age_nz[true_labels_age_nz == year] - year
    MAE = np.mean(np.abs(single_errors))
    years_mae.append(MAE)
plt.figure(figsize=(8, 6))
plt.plot(years, years_mae, color='red', linestyle='-', label='Years MSE')
plt.xlabel('Year')
plt.ylabel('MAE')
plt.title('MAE/Year')
plt.legend()
plt.savefig(f"MS3/Model/{model_type}/plot_mae_year.png")
plt.show()
##
years_mae_positive = []
years_mae_negative = []
for year in years:
    predicted_for_year = predicted_age_nz[true_labels_age_nz == year]
    single_errors = predicted_age_nz[true_labels_age_nz == year] - year
    negative_errors = np.abs(single_errors[single_errors <= 0])
    positive_errors = single_errors[single_errors >= 0]
    years_mae_negative.append(np.mean(negative_errors))
    years_mae_positive.append(np.mean(positive_errors))
plt.figure(figsize=(8, 6))
plt.plot(years, years_mae_positive, color='red', linestyle='-', label='Years MAE Positive')
plt.plot(years, years_mae_negative, color='blue', linestyle='-', label='Years MAE Negative')
plt.xlabel('Years')
plt.ylabel('MAE')
plt.title('Years MAE Split')
plt.legend()
plt.savefig(f"MS3/Model/{model_type}/plot_years_mae_split.png")
plt.show()

##
# https://stackoverflow.com/questions/434287/how-to-iterate-over-a-list-in-chunks#answer-434328
# Display in chunks for better readability
# def chunker(seq, size):
#    return (seq[pos:pos + size] for pos in range(0, len(seq), size))
#
#
# chunk_years = []
# chunk_mae_positive = []
# chunk_mae_negative = []
# for year_chunk in chunker(years, 10):
#    chunk_years.append(max(year_chunk) + min(year_chunk) / 2)
#    chunk_mae_positive.append(np.mean(np.array(years_mae_positive)[year_chunk]))
#    chunk_mae_negative.append(np.mean(np.array(years_mae_negative)[year_chunk]))
# plt.figure(figsize=(8, 6))
# plt.plot(chunk_years, chunk_mae_positive, color='red', linestyle='-', label='Years MAE Positive')
# plt.plot(chunk_years, chunk_mae_negative, color='blue', linestyle='-', label='Years MAE Negative')
# plt.xlabel('Years')
# plt.ylabel('MAE')
# plt.title('Year Chunks MAE Split')
# plt.legend()
# plt.savefig(f"MS3/Model/{model_type}/plot_year_chunks_mae_split.png")
# plt.show()
##
incorrect_predictions_age = np.where(predicted_age_nz != true_labels_age_nz)[0]
print("All", len(true_labels_age_nz))
print("Incorrect", len(incorrect_predictions_age))
print("Quotient", len(incorrect_predictions_age) / len(predicted_age_nz))
exit(0)
##
# -- FIND FALSE PREDICTED IMAGES IN TEST SET ---------------------------------------------------------------------------
# find index of the false predicted image
incorrect_predictions_face = np.where(predicted_face != true_labels_face)[0]
incorrect_predictions_gender = np.where(predicted_gender != true_labels_gender)[0]

# extract path for each image
incorrect_image_paths_face = train_df['path'].iloc[test_size + val_size + incorrect_predictions_face].tolist()
incorrect_image_paths_age = train_df['path'].iloc[test_size + val_size + incorrect_predictions_age].tolist()
incorrect_image_paths_gender = train_df['path'].iloc[test_size + val_size + incorrect_predictions_gender].tolist()
##
# extract false predicted labels and true labels
predicted_labels_face = predicted_face[incorrect_predictions_face]
true_labels_face = true_labels_face[incorrect_predictions_face]
##
predicted_labels_age = predicted_age[incorrect_predictions_age]
true_labels_age = true_labels_age[incorrect_predictions_age]

predicted_labels_gender = predicted_gender[incorrect_predictions_gender]
true_labels_gender = true_labels_gender[incorrect_predictions_gender]

# make dataframes
df_incorrect_paths_face = pd.DataFrame({'Image_Path': incorrect_image_paths_face,
                                        'Predicted_Face': predicted_labels_face,
                                        'True_Face': true_labels_face})

df_incorrect_paths_age = pd.DataFrame({'Image_Path': incorrect_image_paths_age,
                                       'Predicted_Age': predicted_labels_age,
                                       'True_Age': true_labels_age})

df_incorrect_paths_gender = pd.DataFrame({'Image_Path': incorrect_image_paths_gender,
                                          'Predicted_Gender': predicted_labels_gender,
                                          'True_Gender': true_labels_gender})

print('---------------------------------------------------------------------------------------------------------------')
# save to csv
csv_file_face = 'MS3/Model/{}/incorrect_predictions_face.csv'.format(model_type)
df_incorrect_paths_face.to_csv(csv_file_face, index=False)
print(f"Falsch vorhergesagte Bildpfade für Face wurden in '{csv_file_face}' gespeichert.")

csv_file_age = 'MS3/Model/{}/incorrect_predictions_age.csv'.format(model_type)
df_incorrect_paths_age.to_csv(csv_file_age, index=False)
print(f"Falsch vorhergesagte Bildpfade für Age wurden in '{csv_file_age}' gespeichert.")

csv_file_gender = 'MS3/Model/{}/incorrect_predictions_gender.csv'.format(model_type)
df_incorrect_paths_gender.to_csv(csv_file_gender, index=False)
print(f"Falsch vorhergesagte Bildpfade für Gender wurden in '{csv_file_gender}' gespeichert.")
