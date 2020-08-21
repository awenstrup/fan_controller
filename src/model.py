"""Train the model. Looks in the images folder for the training set

Saves the model to the model folder
"""

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

import paths

# Globals

# Create Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

DATA_DIR = pathlib.Path(paths.IMAGES)
WIDTH = 480  # pixels
HEIGHT = 640  # pixels

CLASS_NAMES = ["not_tito", "tito"]  # negative, positive
CLASS_WEIGHTS = {
    CLASS_NAMES.index("tito"): 1,
    CLASS_NAMES.index("not_tito"): 0.1,
}

VALIDATION_SPLIT = 0.2
EPOCHS = 8  # tuned by hand, 


# Helpers
def view_tito() -> None:
    """Display an image from the tito directory to make sure data loaded correctly"""
    tito = list(DATA_DIR.glob(path.join(paths.IMAGES, "tito/*")))
    image = pil.Image.open(str(tito[0]))
    logger.info(image.size)  # Looks to be 480 by 640 for raspi camera
    logger.info(image.mode)
    image.show()

def get_training():
    """Load and return the training dataset"""
    return keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="training",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        class_names=CLASS_NAMES,
    )

def get_validation():
    """Load and return the validation training set"""
    return keras.preprocessing.image_dataset_from_directory(
        DATA_DIR,
        validation_split=VALIDATION_SPLIT,
        subset="validation",
        seed=123,
        image_size=(HEIGHT, WIDTH),
        class_names=CLASS_NAMES,
    )

def get_model():
    """Generate the network architecture

    For now, this is completely unparameterized and tuned by hand, but
    seems to be working fairly well.

    1. Preprocessing layers downsample images and remap pixel values
    2. Convolutional layers determine image properties
    3. Densely connected layers learn from image properties and output result
    """
    model = keras.models.Sequential()

    # PREPROCESSING LAYERS
    # Downsample orignal images to 40x40 images (40, 40, 3)
    model.add(layers.MaxPooling2D((HEIGHT/40, WIDTH/40), input_shape=(HEIGHT, WIDTH, 3)))
    # Remap pixels to [0,1] (40, 40, 3)
    model.add(layers.experimental.preprocessing.Rescaling(1./255))

    # CONVOLUTIONAL LAYERS
    # First covolutional layer (38, 38, 32)
    model.add(layers.Conv2D(32, (3, 3), activation='relu'))
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
    # Output layer (1)
    model.add(layers.Dense(1, activation='sigmoid'))

    logger.debug(model.summary())
    return model

def train_model(model, training, validation):
    """Train ithe model

    Args:
        model: A network to train
        training: a training dataset
        validation: a validation dataset

    Returns:
        The trained model
        The history of the metrics by epoch, for performance evaluation
    """

    model.compile(optimizer="adam",
                loss="binary_crossentropy",
                metrics=[
                    keras.metrics.Precision(), 
                    keras.metrics.Recall(),
                    keras.metrics.BinaryAccuracy(), 
                    keras.metrics.FalsePositives(), 
                    keras.metrics.FalseNegatives(),
                    keras.metrics.TruePositives(),
                    keras.metrics.TrueNegatives(),
                ]
    )

    return model, model.fit(
        training, 
        validation_data=validation, 
        epochs=EPOCHS, 
        class_weight=CLASS_WEIGHTS,
    )

def save_model(model):
    """Save the model"""
    model.save(paths.MODEL)

def save_lite_model(model):
    """Save a lite version of the model to run on RaspPi"""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    if not path.exists(paths.LITE_MODEL):
        os.mkdir(paths.LITE_MODEL)
    
    with tf.io.gfile.GFile(paths.LITE_MODEL + "/lite_model", 'wb') as f:
        f.write(tflite_model)

def load_model():
    """Load a pretrained model"""
    model = keras.models.load_model(paths.MODEL)
    return model

def plot_accuracy(history):
    """Generate some plots for evaluating the model

    Args:
        history: The metrics to evaluate, given by epoch
    
    Note: This model is being used to turn a fan on and off. The cost of turning
    the fan on when the dog isn't there (false positives) is low, nothing really 
    happens. The cost of leaving the fan off when the dog is there (false negatives)
    is high; the dog may learn that the fan doesn't work. Therefore, recall is the 
    most useful metric"""
    # Basic metrics
    plt.plot(history.history['val_binary_accuracy'], label='binary_accuracy')
    plt.plot(history.history['val_precision'], label = 'precision')
    plt.plot(history.history['val_recall'], label = 'recall')  # This is most important for us!
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend(loc='lower right')
    plt.show()

    # # Wrong decisions
    # plt.plot(history.history['val_false_positives'], label='Fan was needlessly on')
    # plt.plot(history.history['val_false_negatives'], label = 'Fan should have turned on')
    # plt.xlabel('Epoch')
    # plt.ylabel('False Results')
    # plt.legend(loc='lower right')
    # plt.show()

    # # Correct decisions
    # plt.plot(history.history['val_true_positives'], label='Fan was off :)')
    # plt.plot(history.history['val_true_negatives'], label = 'Fan was on :)')
    # plt.xlabel('Epoch')
    # plt.ylabel('False Results')
    # plt.legend(loc='lower right')
    # plt.show()

    # Tito was in front of the fan. Was it on? (this is summarized by precision)
    plt.plot(history.history['val_false_negatives'], label = 'Fan should have turned on')
    plt.plot(history.history['val_true_positives'], label = 'Fan was on :)')
    plt.xlabel('Epoch')
    plt.ylabel('Tito in front of fan')
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    t = get_training()
    logger.debug(t.class_names)
    v = get_validation()
    m = load_model()

    # m, h = train_model(m, t, v)
    # plot_accuracy(h)

    save_lite_model(m)
    # save_model(m)
