import tensorflow as tf

def load_and_preprocess_image(image_path):
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return img