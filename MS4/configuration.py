# should be a yaml, but for a project this small this is faster.
# Stores keys like image shape, batch size and such

import os
BATCH_SIZE = 64
EPOCHS = 200  # 100
LEARNING_RATE = 0.001
BUFFER_SIZE = 1000

MODEL_NUMBER = 7

# MODEL_NAME := preprocessing csv name without .csv
MODEL_NAME, IMG_SHAPE = "celeb_a", (178, 218, 3)
# MODEL_NAME, IMG_SHAPE = "promis", (200, 200, 3)
MODEL_IDENTIFIER = f"{MODEL_NAME}_model_{MODEL_NUMBER}_EPOCHS_{EPOCHS}_LR_{LEARNING_RATE}"

FOLDER = "MS4"
OUTPUT = "output"
DATA = "data"
PREPROCESSING = "preprocessing"
PREPROCESSING_CSV = os.path.join(FOLDER, PREPROCESSING, f'{MODEL_NAME}.csv')
BASE_OUTPUT = os.path.join(FOLDER, OUTPUT, MODEL_IDENTIFIER)
MODEL_PATH = os.path.join(BASE_OUTPUT, "siamese_model")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")
TRAINING_HISTORY_PATH = os.path.join(BASE_OUTPUT, "training_history.csv")
LOG_PATH = os.path.join(BASE_OUTPUT, 'ms4.log')
DATA_PATH = os.path.join(FOLDER, DATA)
PAIR_PATH = os.path.join(DATA_PATH, f'pairs_{MODEL_NAME}.csv')
HEAT_MAP_PATH = os.path.join(BASE_OUTPUT, "confusion_matrix.png")
CLASSIFICATION_REPORT_PATH = os.path.join(BASE_OUTPUT, "report.txt")
INFO_PATH = os.path.join(DATA_PATH, f"info_{MODEL_NAME}.txt")
