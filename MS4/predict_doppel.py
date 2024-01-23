import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import configuration
import MS4.utils
import matplotlib.pyplot as plt


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    squared_preds = tf.keras.backend.square(y_pred)
    squared_margin = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
    loss = tf.keras.backend.mean(y_true * squared_preds + (1 - y_true) * squared_margin)
    return loss


# Load your Siamese network model
custom_objects = {"MS4.utils": MS4.utils, "euclidean_distance": MS4.utils.euclidean_distance,
                  "contrastive_loss": contrastive_loss}
model = tf.keras.models.load_model(configuration.MODEL_PATH, custom_objects=custom_objects)

# Load the CSV file
df = pd.read_csv("MS4/data/doppel.csv")

predictions = []

# Iterate through each row in the DataFrame
for index, row in df.iterrows():
    # Load and preprocess images
    img_path1 = row['image1']
    img_path2 = row['image2']

    # Load and preprocess images
    img1 = image.load_img(img_path1, target_size=(178, 218))
    img1 = image.img_to_array(img1)
    img1 = preprocess_input(img1)

    img2 = image.load_img(img_path2, target_size=(178, 218))
    img2 = image.img_to_array(img2)
    img2 = preprocess_input(img2)

    # Expand dimensions to match the input shape expected by the model
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # Make predictions
    prediction = model.predict([img1, img2])
    predictions.append(prediction[0])

    # Interpret the predictions as needed for your task
    print(f"Prediction for pair: {prediction}")

fig, axs = plt.subplots(4, 2, figsize=(15, 5 * 4))
print(predictions)
for index, row in df.iterrows():
    print(row['image1'])
    print(row['image2'])
    axs[index, 0].imshow(image.load_img(row['image1'], target_size=(178, 218)))
    axs[index, 0].set_title("Basis")
    axs[index, 0].axis('off')

    axs[index, 1].imshow(image.load_img(row['image2'], target_size=(178, 218)))
    axs[index, 1].set_title(f"Negativ: {predictions[index]}")
    axs[index, 1].axis('off')

plt.savefig('/tmp/promi_triples.png')
plt.show()
