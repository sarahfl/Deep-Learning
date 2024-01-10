# should be a yaml, but for a project this small this is faster.
# Stores keys like image shape, batch size and such

import os
IMG_SHAPE = (28, 28, 1)
BATCH_SIZE = 64
EPOCHS = 100

FOLDER = "MS4"
OUTPUT = "output"
MODEL_IDENTIFIER = "model1"
BASE_OUTPUT = os.path.join(FOLDER, OUTPUT, MODEL_IDENTIFIER)
MODEL_PATH = os.path.join(BASE_OUTPUT, "siamese_model")
PLOT_PATH = os.path.join(BASE_OUTPUT, "plot.png")

LOG_PATH = os.path.join(BASE_OUTPUT, 'ms4.log')
