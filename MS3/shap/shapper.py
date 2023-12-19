import shap
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np

model_type = "model2_regression"


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


model = tf.keras.models.load_model('MS3/Model/{}/model.keras'.format(model_type))

# Load the Panda file containing all image paths
df = pd.read_csv('MS3/Model/data/Face_regression.csv')  # Replace with your Panda file


# Function to load and preprocess an image using OpenCV
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (200, 200))  # Resize to match model input size
    img = img / 255.0  # Normalize pixel values
    return img


# Choose an index or row number to explain (assuming it's the first row here)
row_index = 2

# Load and preprocess the selected image
image_path = df.iloc[row_index]['path']  # Assuming 'path' is the column containing image paths
print(image_path)
image = load_and_preprocess_image(image_path)
image_for_shap = np.expand_dims(image, axis=0)  # Add batch dimension

# Create a DeepExplainer object
explainer = shap.DeepExplainer(model, np.zeros_like(image_for_shap))  # Background data could be zeros

# Compute SHAP values
shap_values = explainer.shap_values(image_for_shap)

# Visualize SHAP values overlaid on the image
shap.image_plot(shap_values, -image_for_shap)