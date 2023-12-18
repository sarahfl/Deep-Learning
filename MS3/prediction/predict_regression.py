import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import evaluate_predictions

model_type = 'model2_regression'
BATCH_SIZE = 32


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


# load model
tf.keras.utils.get_custom_objects()['CustomMSE'] = CustomMSE
model = tf.keras.models.load_model(f'MS3/Model/{model_type}/model.keras')

# read promi dataset
df = pd.read_csv(f'MS3/prediction/deutschePromis_regression.csv', index_col=0)

# get promi names
promis = df['path']
names = []
for name in promis:
    name_split = name.split('/')
    name_promi = name_split[4]
    names.append(name_promi)
print(names)

# -- MAKE DATASET AND ONE-HOT-ENCODING ---------------------------------------------------------------------------------
age = df['age'].values.astype(int)
one_hot_gender = pd.get_dummies(df['gender']).astype(int)
one_hot_face = pd.get_dummies(df['face']).astype(int)

one_hot_gender = one_hot_gender.to_numpy()
one_hot_face = one_hot_face.to_numpy()

dataset = tf.data.Dataset.from_tensor_slices((df['path'].values, age, one_hot_gender, one_hot_face))
dataset = dataset.map(load_and_preprocess_image)
print('Aufbau des Datensets: ', dataset.element_spec)
dataset = dataset.batch(BATCH_SIZE)

# -- PREDICTION --------------------------------------------------------------------------------------------------------
predictions = model.predict(dataset)

age_array, gender_array, face_array = evaluate_predictions.evaluate(predictions)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.4, wspace=0.35)
for i in range(3):
    for j in range(3):
        index = i * 3 + j

        img = mpimg.imread(promis[index])
        axs[i, j].imshow(img)
        axs[i, j].axis('off')

        # title
        title = f"Age: {age_array[index]}, Gender: {gender_array[index]}, Face: {face_array[index]}"
        axs[i, j].set_title(title)
plt.savefig('MS3/Model/{}/promis_regression.png'.format(model_type))
plt.show()
