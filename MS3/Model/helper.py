import tensorflow as tf
def load_and_preprocess_image(image_path, label_age, label_gender, label_face):
    img = tf.image.decode_jpeg(tf.io.read_file(image_path), channels=3)
    return img, {'age_output': label_age, 'gender_output': label_gender, 'face_output': label_face}

def get_label(label):
    return {'age': tf.reshape(tf.keras.backend.cast(label[0], tf.keras.backend.floatx()), (-1, 1)),
            'gender': tf.reshape(tf.keras.backend.cast(label[1], tf.keras.backend.floatx()), (-1, 1)),
            'face': tf.reshape(tf.keras.backend.cast(label[2], tf.keras.backend.floatx()), (-1, 1))}

def decode_img(img_path):
    image_size = (200, 200)
    img = tf.io.read_file(img_path)
    img = tf.image.decode_image(
        img, channels=3, expand_animations=False
    )
    img.set_shape((image_size[0], image_size[1], 3))
    return img

def process_path(file_path, labels):

    label = get_label(labels)
    img = decode_img(file_path)
    return img, label