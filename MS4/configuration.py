# should be a yaml, but for a project this small this is faster.
# Stores keys like image shape, batch size and such

import os
BATCH_SIZE = 64
EPOCHS = 5  # 100
LEARNING_RATE = 0.001

MODEL_NUMBER = 2

# MODEL_NAME := preprocessing csv name without .csv
# MODEL_NAME, IMG_SHAPE = "celeb_a", (178, 218, 3)
MODEL_NAME, IMG_SHAPE = "promis", (200, 200, 3)
MODEL_IDENTIFIER = f"{MODEL_NAME}_model_{MODEL_NUMBER}_EPOCHS_{EPOCHS}_LR_{LEARNING_RATE}"

FOLDER = "MS4"
OUTPUT = "output"
PREPROCESSING = "preprocessing"
PREPROCESSING_CSV = os.path.join(FOLDER, PREPROCESSING, f'{MODEL_NAME}.csv')
BASE_OUTPUT = os.path.join(FOLDER, OUTPUT, MODEL_IDENTIFIER)
MODEL_PATH = os.path.join(BASE_OUTPUT, "siamese_model")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
TRAINING_HISTORY_PATH = os.path.join(BASE_OUTPUT, "training_history.csv")
LOG_PATH = os.path.join(BASE_OUTPUT, 'ms4.log')
PAIR_PATH = f'MS4/data/pairs_{MODEL_NAME}.csv'
