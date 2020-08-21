"""Script to be run on Raspberry Pi
Periodically takes an image (with attached camera), 
and updates the fan state based on the model's prediction

I don't really want label any more data, so I am not adding
these images to a new training set.
"""
# Imports
import sys
import time

# Exptended Python
import cv2
import boto3
import gpiozero
import numpy as np
import tflite_runtime.interpreter as tflite

sys.path.append("/home/pi/fan_controller/src")
import paths

# Globals
PERIOD = 10  # 10 seconds
PIN = 17


def load_model():
    """Load a pretrained model"""
    model = tflite.Interpreter(paths.LITE_MODEL + "/lite_model")
    model.allocate_tensors()
    return model


def predict(model, image):
    """Run an image through the model"""
    model.set_tensor(model.get_input_details()[0]["index"], image)
    model.invoke()
    return model.get_tensor(model.get_output_details()[0]["index"])


def update(model, fan_state, fan):
    """Given a model, take a photo and update the fan state"""
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    _, image = camera.read()
    del camera

    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image = np.asarray([image], dtype=np.float32)

    prediction = predict(model, image)
    print(prediction)
    if abs(prediction - 0.5) > 0.1:
        # Only update model if the prediction is >60% confident to reduce random toggling
        temp = True if prediction > 0.5 else False
        if temp is not fan_state:
            fan_state = temp
            toggle_fan()

    print("Fan state: " + str(fan_state))
    return fan_state


def toggle_fan():
    if fan_state:
        fan.on()
    else:
        fan.off()


if __name__ == "__main__":
    fan_state = False  # False is off, True is on
    fan = gpiozero.DigitalOutputDevice(PIN)
    m = load_model()

    while True:
        time.sleep(PERIOD)
        try:
            fan_state = update(m, fan_state, fan)
        except Exception as e:
            print(e)
            fan.off()
            break
