import configuration
import tensorflow as tf
import pandas as pd
import MS4.utils
import helper
import logging
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

pair_df = pd.read_csv(configuration.PAIR_PATH)
pair_1 = pair_df['image1'].to_numpy()
pair_2 = pair_df['image2'].to_numpy()
labels = pair_df['PairLabels'].to_numpy()

dataset = tf.data.Dataset.from_tensor_slices(((pair_1, pair_2), labels))
dataset = dataset.map(lambda pair, label: helper.load_images(pair[0], pair[1], label))

dataset_size = len(pair_df)
train_size = int(0.8 * dataset_size)
val_size = int(0.1 * dataset_size)
test_size = dataset_size - train_size - val_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size).take(val_size)
test_dataset = dataset.skip(train_size + val_size)

train_dataset = train_dataset.batch(configuration.BATCH_SIZE)
val_dataset = val_dataset.batch(configuration.BATCH_SIZE)
test_dataset = test_dataset.batch(configuration.BATCH_SIZE)


def contrastive_loss(y_true, y_pred, margin=1):
    y_true = tf.cast(y_true, y_pred.dtype)
    squared_preds = tf.keras.backend.square(y_pred)
    squared_margin = tf.keras.backend.square(tf.keras.backend.maximum(margin - y_pred, 0))
    loss = tf.keras.backend.mean(y_true * squared_preds + (1 - y_true) * squared_margin)
    return loss


custom_objects = {"MS4.utils": MS4.utils, "euclidean_distance": MS4.utils.euclidean_distance,
                  "contrastive_loss": contrastive_loss}
logging.info(f"Loading model {configuration.MODEL_PATH}")
model = tf.keras.models.load_model(configuration.MODEL_PATH, custom_objects=custom_objects)

dataset_to_test = test_dataset

predictions = model.predict(dataset_to_test)
# np.save("test.anp", predictions)
# predictions = np.load("test.anp.npy")
# Assuming your predictions are in the range [0, 1] and you want to round them to 0 or 1
true_labels = []

for _, labels in dataset_to_test:
    flatten_labels = labels.numpy().flatten()
    true_labels = np.concatenate((true_labels, flatten_labels))

print(len(predictions))
print("---")
print(len(true_labels))
# print(sum(predictions - true_labels))
predictions_round_int = np.round(predictions.flatten())
predictions_round_int = predictions_round_int.astype(int)
# Create confusion matrix
conf_matrix = confusion_matrix(true_labels, predictions_round_int)

# Print classification report
report = classification_report(true_labels, predictions_round_int)
print(report)
with open(configuration.CLASSIFICATION_REPORT_PATH, 'w') as file:
    print(report, file=file)

# Plot confusion matrix as a heatmap
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Predicted 0', 'Predicted 1'],
            yticklabels=['Actual 0', 'Actual 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.savefig(configuration.HEAT_MAP_PATH)
plt.show()
