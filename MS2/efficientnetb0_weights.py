##
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

# -- EfficientNet B0 WEIGHTS -------------------------------------------------------------------------------------------
##
train_dir = '/home/sarah/Deep-Learning/Train_Test_Folder_2/train'
test_dir = ('/home/sarah/Deep-Learning/Train_Test_Folder_2/test')

BATCH_SIZE = 32
IMG_SIZE = (200, 200)
EPOCHS = 5

##
# -- TRAINING AND VALIDATION DATA --------------------------------------------------------------------------------------
train_dataset, val_dataset = tf.keras.utils.image_dataset_from_directory(train_dir,
                                                                         labels='inferred',
                                                                         label_mode='binary',
                                                                         class_names=None,
                                                                         color_mode='rgb',
                                                                         batch_size=BATCH_SIZE,
                                                                         image_size=IMG_SIZE,
                                                                         shuffle=True,
                                                                         seed=100,
                                                                         validation_split=0.17,
                                                                         subset='both')

##
# -- TEST DATA ---------------------------------------------------------------------------------------------------------
test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                           labels='inferred',
                                                           label_mode='binary',
                                                           class_names=None,
                                                           color_mode='rgb',
                                                           batch_size=BATCH_SIZE,
                                                           image_size=IMG_SIZE,
                                                           shuffle=True)
##
class_names = train_dataset.class_names
print(class_names)

##
# plot example images from dataset with label 0=face, 1=noFace
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(labels[i].numpy().astype('uint8'))
        plt.axis("off")
plt.show()

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
for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')  #
plt.show()

##
# -- CREATE BASE MODEL -------------------------------------------------------------------------------------------------
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.EfficientNetB0(
    weights='imagenet',  # Load weights pre-trained on ImageNet.
    input_shape=IMG_SHAPE,
    include_top=False,
)

base_model.trainable = False
print("Number of layers in the base model: ", len(base_model.layers))

##
# -- RETRAIN TOP LAYERS ------------------------------------------------------------------------------------------------
fine_tune_at = 100
# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

##
# -- CREATE NEW MODEL ON TOP -------------------------------------------------------------------------------------------
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

inputs = tf.keras.Input(shape=IMG_SHAPE)
x = data_augmentation(inputs)
# Pre-trained Model weights requires that input be scaled from (0, 255) to a range of [-1,1]
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)  # Regularize with dropout
outputs = tf.keras.layers.Dense(1)(x)
model = tf.keras.Model(inputs, outputs)

model.summary()
print('Number of trainable weights={}'.format(len(model.trainable_weights)))

##
# -- COMPILE THE MODEL -------------------------------------------------------------------------------------------------
base_learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[tf.keras.metrics.BinaryAccuracy(threshold=0, name='accuracy')])

##
# save model summary to file
with open('summaryModel_weights_2_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(base_learning_rate, EPOCHS, BATCH_SIZE), 'w') as f:
    model.summary(print_fn=lambda x: f.write(x + '\n'))

# tf.keras.utils.plot_model(model, show_shapes=True, to_file='layersModel_weights_LR_{}_EPOCHS_{}_BATCH_{}.png'.format(base_learning_rate, EPOCHS, BATCH_SIZE))  # save model as png

##
# -- TRAIN THE MODEL ---------------------------------------------------------------------------------------------------
history = model.fit(train_dataset,
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    validation_data=validation_dataset)

##
hist_df = pd.DataFrame(history.history)
hist_csv_file = 'historyModel_weights_2_LR_{}_EPOCHS_{}_BATCH_{}.csv'.format(base_learning_rate, EPOCHS, BATCH_SIZE)
with open(hist_csv_file, mode='w') as f:
    hist_df.to_csv(f)

##
# save the model
model.save('/home/sarah/Deep-Learning/MS2/EfficientNetB0/weights_2/model_100.keras')

##
# -- REVIEW THE LEARNING CURVES ----------------------------------------------------------------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(8, 8))
plt.subplot(2, 1, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.ylabel('Accuracy')
plt.ylim([min(plt.ylim()), 1])
plt.title('Training and Validation Accuracy')

plt.subplot(2, 1, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.ylabel('Cross Entropy')
plt.ylim([0, 1.0])
plt.title('Training and Validation Loss')
plt.xlabel('epoch')
# save plot
plt.savefig('plotModel_weights_2_LR_{}_EPOCHS_{}_BATCH_{}.png'.format(base_learning_rate, EPOCHS, BATCH_SIZE))
plt.show()

##
# -- EVALUATE VALIDATION AND TEST DATA ---------------------------------------------------------------------------------
lossV, accuracyV = model.evaluate(validation_dataset)
lossT, accuracyT = model.evaluate(test_dataset)
print('Validation accuracy :', accuracyV)
print('Validation Loss :', lossV)
print('Test accuracy :', accuracyT)
print('Test Loss :', lossT)

##
# save evaluation to file
dict = {'validation accuracy': accuracyV, 'validation loss': lossV, 'test accuracy': accuracyT, 'test loss': lossT}
f = open('evaulationModel_weights_2_LR_{}_EPOCHS_{}_BATCH_{}.txt'.format(base_learning_rate, EPOCHS, BATCH_SIZE), 'w')
f.write('dict = ' + repr(dict) + '\n')
f.close()

##
# plot example from test image with predicted labels
image_batch, label_batch = test_dataset.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch).flatten()

# Apply a sigmoid since our model returns logits
predictions = tf.nn.sigmoid(predictions)
predictions = tf.where(predictions < 0.5, 0, 1)

print('Predictions:\n', predictions.numpy())
print('Labels:\n', label_batch)

plt.figure(figsize=(10, 10))
for i in range(9):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image_batch[i].astype("uint8"))
    plt.title(class_names[predictions[i]])
    plt.axis("off")
plt.savefig('predictions_2.png')
plt.show()
