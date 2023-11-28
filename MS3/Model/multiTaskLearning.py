##
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.callbacks import EarlyStopping


##
# -- GET DATA




train_dir = '/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/train'
val_dir = '/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/val'
test_dir = '/home/sarah/Deep-Learning/MS3/preprocessing/Train_Test_Folder/test'

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 50
num_age_classes = 7

##
# -- TRAINING DATA -----------------------------------------------------------------------------------------------------
train_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                            labels='inferred',
                                                            label_mode='categorical',
                                                            class_names=None,
                                                            color_mode='rgb',
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            shuffle=True,
                                                            )
##
# -- VALIDATION DATA ---------------------------------------------------------------------------------------------------
validation_dataset = tf.keras.utils.image_dataset_from_directory(val_dir,
                                                                 labels='inferred',
                                                                 label_mode='categorical',
                                                                 class_names=None,
                                                                 color_mode='rgb',
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 shuffle=True,
                                                                 )
##
# -- TEST DATA ---------------------------------------------------------------------------------------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           labels='inferred',
                                                           label_mode='categorical',
                                                           class_names=None,
                                                           color_mode='rgb',
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE,
                                                           shuffle=True,
                                                           )
##
class_names = train_dataset.class_names
print(class_names)

##
# plot example images from dataset with label 0=face, 1=noFace
plt.figure(figsize=(20, 20))
for images, labels in train_dataset.take(1):
    for i in range(25):
        ax = plt.subplot(5, 5, i + 1)

        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy().astype('uint8'))
        plt.axis("off")
plt.show()

##
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

##
# -- DATA AUGMENTATION -------------------------------------------------------------------------------------------------
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip('horizontal'),
    tf.keras.layers.RandomRotation(0.2)
])
##
# -- CREATE BASE MODEL -------------------------------------------------------------------------------------------------
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=IMG_SHAPE,
    include_top=False,
)

base_model.summary()
print(len(base_model.trainable_variables))

base_model.trainable = False

##
# -- CREATE NEW MODEL ON TOP -------------------------------------------------------------------------------------------
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
# Pre-trained Model weights requires that input be scaled from (0, 255) to a range of [-1,1]
x = preprocess_input(x)
x = base_model(x)

x = tf.keras.layers.Dense(128, activation='relu')(x)

# age prediction
age_prediction = tf.keras.layers.Dense(num_age_classes, name='age_output', activation='softmax')(x)

# gender prediction
gender_prediction = tf.keras.layers.Dense(1, name='gender_output', activation='sigmoid')(x)

model = tf.keras.Model(inputs, [age_prediction, gender_prediction])

model.summary()
print("Trainable Variables: ", len(model.trainable_variables))

##
# -- COMPILE THE MODEL -------------------------------------------------------------------------------------------------
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss={'age_output': tf.keras.losses.CategoricalCrossentropy(),
                    'gender_output': tf.keras.losses.BinaryCrossentropy()},
              metrics={'age_output': 'accuracy', 'gender_output': 'accuracy'})

##
# save model summary to file
with open('model_summary_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(base_learning_rate, EPOCHS, BATCH_SIZE), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

##
# -- TRAIN THE MODEL ---------------------------------------------------------------------------------------------------
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=validation_dataset)

##
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

hist_df = pd.DataFrame(history.history)
hist_csv_file = 'history.csv'
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

# save the model
model.save('/home/sarah/Deep-Learning/MS3/model.keras')
