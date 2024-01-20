from keras import metrics
from tensorflow.keras import metrics
from tensorflow.keras.models import Model
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import csv
import matplotlib.image as mpimg


class SiameseModel(Model):
    """
    A custom keras model
    https://keras.io/examples/vision/siamese_network/#summary
    """

    def __init__(self, siamese_network, margin=0.9):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin
        self.loss_tracker = metrics.Mean(name="loss")
        self.accuracy = metrics.BinaryAccuracy(name="accuracy")

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        with tf.GradientTape() as tape:
            loss = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    def test_step(self, data):
        loss = self._compute_loss(data)

        # Get predictions
        ap_distance, an_distance = self.siamese_network(data)
        predictions = tf.sigmoid(ap_distance - an_distance)

        # Let's update and return the loss metric.
        self.loss_tracker.update_state(loss)
        self.accuracy.update_state(tf.ones_like(predictions), tf.cast(predictions > 0.5, tf.float32))

        return {"loss": self.loss_tracker.result(), "accuracy": self.accuracy.result()}

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)

        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)

        predictions = tf.sigmoid(ap_distance - an_distance)
        self.accuracy.update_state(tf.ones_like(predictions), tf.cast(predictions > 0.5, tf.float32))

        return loss

    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker, self.accuracy]


class DistanceLayer(layers.Layer):
    """
    A custom keras layer to calculate the distance between (anchor,positive) and (anchor, negative)
    https://keras.io/examples/vision/siamese_network/#summary
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def call(self, inputs):
        anchor, positive, negative = inputs
        anchor = tf.math.l2_normalize(anchor, axis=1)
        positive = tf.math.l2_normalize(positive, axis=1)
        negative = tf.math.l2_normalize(negative, axis=1)

        ap_distance = tf.reduce_sum(tf.square(anchor - positive), -1)
        an_distance = tf.reduce_sum(tf.square(anchor - negative), -1)
        return (ap_distance, an_distance)


def get_model(input_shape):
    """
    Build a custom model on basis of MobilenetV2 to use it as feature-extractor
    :param input_shape: input shape of the images
    :return: keras model for encoding
    """
    pretrained_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        weights='imagenet',
        include_top=False
    )

    for i in range(len(pretrained_model.layers) - 27):
        pretrained_model.layers[i].trainable = False

    encode_model = tf.keras.Sequential([
        pretrained_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(512, activation='relu'),
        #layers.BatchNormalization(),
        #layers.Dropout(0.5),
        layers.Dense(256, activation="relu")
    ], name="Encode_Model")
    return encode_model


def load_and_preprocess_image(image_path):
    """
    Load images from file path and transform the into a tensor
    :param image_path: path to image
    :return: tensor with image content
    """
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return img


def evaluate_model(model, val_dataset):
    """
    Evaluate how the model performs on a validation dataset
    :param model: keras model
    :param val_dataset: validation dataset
    :return: metrics on how the model performed in training
    """

    pos_scores, neg_scores = [], []

    for batch in val_dataset:
        anchor, pos, neg = batch
        pred = model([anchor, pos, neg])
        pos_scores += list(pred[0].numpy())
        neg_scores += list(pred[1].numpy())

    acc = np.sum(np.array(pos_scores) < np.array(neg_scores)) / len(pos_scores)
    means = (np.mean(pos_scores), np.mean(neg_scores))
    stds = (np.std(pos_scores), np.std(neg_scores))
    mins = (np.min(pos_scores), np.min(neg_scores))
    maxs = (np.max(pos_scores), np.max(neg_scores))

    print("Acc:", acc, "Means:", means, "stds:", stds)
    metrics = [acc, means, stds, mins, maxs]
    return metrics


def train_model(model, train_dataset, num_epochs, val_dataset, BATCH_SIZE):
    """
    Custom training loop
    :param model: keras model
    :param train_dataset: training dataset
    :param num_epochs: number of epochs
    :param val_dataset: validation dataset
    :param BATCH_SIZE: batch size
    :return: training loss and metrics per epoch
    """
    metrics = []
    training_loss = []

    for epoch in range(num_epochs):
        losses = []

        for batch in train_dataset:
            anchor, pos, neg = batch
            loss = model.train_on_batch([anchor, pos, neg]) / BATCH_SIZE
            losses.append(loss)

        epoch_loss = np.mean(losses)
        print("Epoch:", epoch, "Loss:", epoch_loss)
        training_loss.append(epoch_loss)

        # Get metrics on test data
        metrics.append(evaluate_model(model, val_dataset))

        # Update learning rate
        if epoch == 3:
            print("LR now 1e-5.")
            model.optimizer.lr.assign(1e-5)
        if epoch == 7:
            print("LR now 1e-6.")
            model.optimizer.lr.assign(1e-6)

    return training_loss, metrics


def plot_metrics(loss, metrics, name):
    """
    Plot the metrics of the performance of the model
    :param loss: loss
    :param metrics: metrics
    :param name: name of model
    :param name: name of model
    """
    n = len(metrics)
    acc = np.zeros(n)
    pmean, pstd, pmin, pmax = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)
    nmean, nstd, nmin, nmax = np.zeros(n), np.zeros(n), np.zeros(n), np.zeros(n)

    for i, data in enumerate(metrics):
        acc[i] = data[0]
        pmean[i] = data[1][0]
        nmean[i] = data[1][1]
        pstd[i] = data[2][0]
        nstd[i] = data[2][1]
        pmin[i] = data[3][0]
        nmin[i] = data[3][1]
        pmax[i] = data[4][0]
        nmax[i] = data[4][1]

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(loss, 'r', label='Loss')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(acc, 'g', label='Accuracy')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.savefig('/home/sarah/Deep-Learning/MS4/triples/Model/{}/accuracy_loss.png'.format(name))
    plt.show()

    plt.figure(figsize=(15, 5))
    plt.title("Separation")
    plt.xlabel("Epochs")
    plt.ylabel("Distance")
    plt.grid()
    epochs = np.arange(n)
    plt.fill_between(epochs, pmin, pmax, alpha=0.1, color="g", label='Positive')
    plt.fill_between(epochs, nmin, nmax, alpha=0.1, color="r", label='Negative')
    plt.fill_between(epochs, pmean - pstd, pmean + pstd, alpha=0.9, color="g")
    plt.fill_between(epochs, nmean - nstd, nmean + nstd, alpha=0.9, color="r")
    plt.legend()
    plt.savefig('/home/sarah/Deep-Learning/MS4/triples/Model/{}/distance.png'.format(name))
    plt.show()


def save_metrics_to_csv(loss, metrics, filename):
    """
    Save metrics to csv
    :param loss: trainings loss
    :param metrics: trainings metrics
    :param filename: filename
    """
    header = ["Epoch", "Training Loss", "Accuracy", "Positive Mean", "Negative Mean", "Positive Std", "Negative Std",
              "Positive Min", "Negative Min", "Positive Max", "Negative Max"]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)

        for epoch, (acc, means, stds, mins, maxs) in enumerate(metrics):
            row = [epoch, loss[epoch], acc, means[0], means[1], stds[0], stds[1], mins[0], mins[1], maxs[0], maxs[1]]
            writer.writerow(row)


def calculate_similarity(encoder_model, image_triples):
    """
    Calculate cosine similarity between (anchor, positive) and (anchor, negative) -> for images
    :param encoder_model: model used for feature extraction
    :param image_triples: list of pathes to images
    :return: List of Triples (anchor, positive_similarity, negative_similarity)
    """
    similarities = []

    for triple in image_triples:
        image1_path, image2_path, image3_path = triple
        image1 = load_and_preprocess_image(image1_path)
        image2 = load_and_preprocess_image(image2_path)
        image3 = load_and_preprocess_image(image3_path)

        # Änderung des Datentyps auf float32 vor dem Preprocessing
        image1 = tf.image.convert_image_dtype(image1, dtype=tf.float32)
        image2 = tf.image.convert_image_dtype(image2, dtype=tf.float32)
        image3 = tf.image.convert_image_dtype(image3, dtype=tf.float32)

        # Preprocess and encode images using the trained encoder
        prep = tf.keras.applications.mobilenet_v2.preprocess_input
        encoded_image1 = encoder_model.predict(np.expand_dims(prep(image1), axis=0))
        encoded_image2 = encoder_model.predict(np.expand_dims(prep(image2), axis=0))
        encoded_image3 = encoder_model.predict(np.expand_dims(prep(image3), axis=0))

        # Calculate cosine similarity between the encoded images
        # similarity = tf.keras.metrics.CosineSimilarity()(encoded_image1, encoded_image2).numpy()
        # similarities.append(similarity)

        cosine_similarity = metrics.CosineSimilarity()

        positive_similarity = cosine_similarity(encoded_image1, encoded_image2)
        # print("Positive similarity:", positive_similarity.numpy())

        negative_similarity = cosine_similarity(encoded_image1, encoded_image3)
        # print("Negative similarity", negative_similarity.numpy())

        similarities.append(('anchor', positive_similarity, negative_similarity))

    return similarities


def calculate_similarity_tensor(encoder_model, test_dataset):
    """
    Calculate cosine similarity between (anchor, positive) and (anchor, negative) -> for tensors
    :param encoder_model: model used for feature extraction
    :param test_dataset: test dataset containing image tensors
    :return: List of Triples (anchor, positive_similarity, negative_similarity)
    """
    similarities = []

    prep = tf.keras.applications.mobilenet_v2.preprocess_input
    cosine_similarity = tf.keras.metrics.CosineSimilarity()

    for anchor_img, positive_img, negative_img in test_dataset:
        anchor_embedding = encoder_model(prep(tf.cast(anchor_img, tf.float32)))
        positive_embedding = encoder_model(prep(tf.cast(positive_img, tf.float32)))
        negative_embedding = encoder_model(prep(tf.cast(negative_img, tf.float32)))
        # print(anchor_img.shape)
        # print(anchor_embedding.shape)
        # Preprocess and encode images using the trained encoder
        encoded_anchor = encoder_model.predict(anchor_img)
        encoded_positive = encoder_model.predict(positive_img)
        encoded_negative = encoder_model.predict(negative_img)

        # Calculate cosine similarity between the encoded images
        positive_similarity = cosine_similarity(encoded_anchor, encoded_positive)
        # print("Positive similarity:", positive_similarity.numpy())

        negative_similarity = cosine_similarity(encoded_anchor, encoded_negative)
        # print("Negative similarity", negative_similarity.numpy())

        similarities.append(('anchor', positive_similarity.numpy(), negative_similarity.numpy()))

    return similarities


def plot_image_triples(image_triples, titles_list, name):
    """
    Plot images for testing
    :param image_triples: List of images containing (anchor_image, positive_image, negative_image)
    :param titles_list: List of results from training containing ('anchor', positive_similarity, negative_similarity)
    :param name: name of the model
    """
    fig, axs = plt.subplots(len(image_triples), 3, figsize=(15, 5 * len(image_triples)))

    for i, ((img1, img2, img3), (title1, title2, title3)) in enumerate(zip(image_triples, titles_list)):
        axs[i, 0].imshow(mpimg.imread(img1))
        axs[i, 0].set_title(title1)
        axs[i, 0].axis('off')

        axs[i, 1].imshow(mpimg.imread(img2))
        axs[i, 1].set_title(f"Positive: {title2}")
        axs[i, 1].axis('off')

        axs[i, 2].imshow(mpimg.imread(img3))
        axs[i, 2].set_title(f"Negative: {title3}")
        axs[i, 2].axis('off')

    plt.savefig('/home/sarah/Deep-Learning/MS4/triples/Model/{}/doppelganger_triples.png'.format(name))
    plt.show()


def convert_to_tuples(triples_list):
    """
    Convert a triple list to a tuple list by taking 1st and 2nd entry of each triple -> for integers
    :param triples_list: list of triples
    :return: list of tuples
    """
    result_list = [(t[1].numpy(), t[2].numpy()) for t in triples_list]
    return result_list


def convert_to_tuples_tensor(triples_tensor):
    """
    Convert a triple list to a tuple list by taking 1st and 2nd entry of each triple -> for tensors
    :param triples_tensor: list of triples
    :return: list of tuples
    """
    result_list = [(t[1], t[2]) for t in triples_tensor]
    return result_list


def calculate_differences_within_tuples(tuple_list):
    """
    Calculate the difference between the 0th and the 1st entry of each tuple
    :param tuple_list: list of tuples
    :return: list of integers
    """
    differences = [(t[0] - t[1]) for t in tuple_list]
    return differences


def average(number_list):
    """
    Calculate the average value of a list of numbers
    :param number_list: list of integers
    :return: average value
    """
    sum_all = sum(number_list)
    final_average = sum_all / len(number_list)

    return final_average


image_pairs = [
    ('/home/sarah/Deep-Learning/MS4/data/doppelganger/SigourneyWeaver_5.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/SigourneyWaver_p_5.jpeg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/SusanSarandon_5.jpg'),
    ('/home/sarah/Deep-Learning/MS4/data/doppelganger/JeffreyDeanMorgan_6.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/JeffryDeanMorgan_p_6.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/JavierBardem_6.jpg'),
    ('/home/sarah/Deep-Learning/MS4/data/doppelganger/ZachBraff_7.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/ZachBraff_p_7.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/DaxShepard_7.jpg'),
    ('/home/sarah/Deep-Learning/MS4/data/doppelganger/NataliePortman_1.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/NataliePortman_p_1.jpg',
     '/home/sarah/Deep-Learning/MS4/data/doppelganger/KeiraKnightley_1.jpg')
]
