"""Train the model. Looks in the ../images folder for the training set"""

# Imports

# Base python
import os
from os import path
import logging

# Extended Python
import numpy as np 
import PIL as pil
import tensorflow as tf 
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import pathlib
import matplotlib.pyplot as plt

# Globals

# Create Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DATA_DIR = pathlib.Path(path.abspath("../images"))
WIDTH = 480
HEIGHT = 640

# Helpers
def view_tito(): 
    tito = list(DATA_DIR.glob('../images/tito/*'))
    image = pil.Image.open(str(tito[0]))
    logger.info(image.size)  # Looks to be 480 by 640 for raspi camera
    logger.info(image.mode)
    image.show()

def get_training():
    return keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.15,
        subset="training",
        seed=123,
        image_size=(HEIGHT, WIDTH),
    )

def get_validation():
    return keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=0.15,
        subset="validation",
        seed=123,
        image_size=(HEIGHT, WIDTH),
    )

def get_model():
    model = keras.models.Sequential()
    # Downsample orignal images to 40x40 images (40, 40, 3)
    model.add(layers.MaxPooling2D((HEIGHT/40, WIDTH/40), input_shape=(HEIGHT, WIDTH, 3)))

    # Remap pixels to [0,1] (40, 40, 3)
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    # CONVOLUTIONAL LAYERS
    # First covolutional layer (38, 38, 32)
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(HEIGHT, WIDTH, 3)))
    # Downsample again (19, 19, 32)
    model.add(layers.MaxPooling2D((2, 2)))
    # Second convolutional layer (17, 17, 64)
    model.add(layers.Conv2D(48, (3, 3), activation='relu'))
    # Downsample again (8, 8, 64)
    model.add(layers.MaxPooling2D((2, 2)))
    # Third convolutional layer (6, 6, 64)
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))

    # DENSE LAYERS
    # Flatten to 1D vector (2304)
    model.add(layers.Flatten())
    # First dense layer (128)
    model.add(layers.Dense(64, activation='relu'))
    # Second dense layer, output (2)
    model.add(layers.Dense(2))

    logger.info(model.summary())
    return model

def train_model(model, training, validation, save=False):
    model.compile(optimizer='adam',
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy']
    )

    return model.fit(
        training, 
        validation_data=validation, 
        epochs=3, 
        callbacks=[save_model] if save else [],
    )

def save_model():
    tf.keras.callbacks.ModelCheckpoint(
        filepath="../model",
        save_weights_only=True,
        verbose=1,
    )

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    # view_tito()
    t = get_training()
    v = get_validation()
    m = get_model()

    h = train_model(m, t, v)
    plot_accuracy(h)

