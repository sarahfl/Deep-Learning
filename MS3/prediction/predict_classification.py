import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

modelType = 'model11_classification'
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

# revert age classification
age_array = []
for age in predicted_age_labels:
    age_range = ''
    if age == 0:
        age_range = '[1..2]'
    elif age == 1:
        age_range = '[3..9]'
    elif age == 2:
        age_range = '[10..20]'
    elif age == 3:
        age_range = '[21..27'
    elif age == 4:
        age_range = '[28..45]'
    elif age == 5:
        age_range = '[46..65]'
    elif age == 6:
        age_range = '[66..116]'
    elif age == 7:
        age_range = 'no age'

    age_array.append(age_range)

# revert face classification
face_array = []
for face in predicted_face_labels:
    fa = ''
    if face == 0 :
        fa = 'yes'
    else:
        fa='no'
    face_array.append(fa)

# revert gender classification
gender_array = []
for gender in predicted_gender_labels:
    ge = ''
    if gender == 0:
        ge = 'm'
    elif gender == 1:
        ge = 'f'
    else:
        ge = 'no'
    gender_array.append(ge)

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
plt.savefig('/home/sarah/Deep-Learning/MS3/Model/{}/promis_classification.png'.format(modelType))
plt.show()
