import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

modelType = 'model1_classification'
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

# load model
model = tf.keras.models.load_model('/home/sarah/Deep-Learning/MS3/Model/{}/model.keras'.format(modelType))

# read promi dataset
df = pd.read_csv('/home/sarah/Deep-Learning/MS3/prediction/deutschePromis_classification.csv', index_col=0)

# get promi names
promis = df['path']
names = []
for name in promis:
    name_split = name.split('/')
    name_promi = name_split[7]
    names.append(name_promi)
print(names)

# -- MAKE DATASET AND ONE-HOT-ENCODING ---------------------------------------------------------------------------------
one_hot_age = pd.get_dummies(df['age']).astype(int)
one_hot_gender = pd.get_dummies(df['gender']).astype(int)
one_hot_face = pd.get_dummies(df['face']).astype(int)

one_hot_age = one_hot_age.to_numpy()
one_hot_gender = one_hot_gender.to_numpy()
one_hot_face = one_hot_face.to_numpy()

dataset = tf.data.Dataset.from_tensor_slices((df['path'].values, one_hot_age, one_hot_gender, one_hot_face))
dataset = dataset.map(load_and_preprocess_image)
print('Aufbau des Datensets: ', dataset.element_spec)
dataset = dataset.batch(BATCH_SIZE)

# -- PREDICTION --------------------------------------------------------------------------------------------------------
predictions = model.predict(dataset)

# extract prediction from every output channel
predictions_age = predictions[0]
predictions_gender = predictions[1]
predictions_face = predictions[2]

# -- PLOT PREDICTION ---------------------------------------------------------------------------------------------------
predicted_age_labels = np.argmax(predictions_age, axis=1)
predicted_gender_labels = np.argmax(predictions_gender, axis=1)
predicted_face_labels = np.argmax(predictions_face, axis=1)

fig, axs = plt.subplots(3, 3, figsize=(10, 10))
for i in range(3):
    for j in range(3):
        index = i * 3 + j

        img = mpimg.imread(promis[index])
        axs[i, j].imshow(img)
        axs[i, j].axis('off')

        # title
        title = f"Age: {predicted_age_labels[index]}, Gender: {predicted_gender_labels[index]}, Face: {predicted_face_labels[index]}"
        axs[i, j].set_title(title)
plt.savefig('/home/sarah/Deep-Learning/MS3/Model/{}/promis_classification.png'.format(modelType))
plt.show()
