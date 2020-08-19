"""Script to be run on Raspberry Pi
Periodically takes an image (with attached camera), 
and uploads it to an S3 bucket. 

I don't really want label any more data, so I am not adding
these images to a new training set.
"""
# Imports

# Exptended Python
import cv2
import boto3
import gpiozero

# Globals
PERIOD = 10  # 10 seconds
PIN = 17

fan_state: bool = False  # False is off, True is on
fan = gpiozero.DigitalOutputDevice(17)

def load_model():
    """Load a pretrained model"""
    model = keras.models.load(paths.MODEL)
    return model

def update(model):
    """Given a model, take a photo and update the fan state"""
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    _, image = camera.read()
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    del(camera)

    prediction = model.predict(image)[0]  # Get only element from list
    if abs(prediction - 0.5) > 0.1:  
        # Only update model if the prediction is >60% confident to reduce random toggling
        temp = True if prediction > 0.5 else False
        if temp is not fan_state:
            fan_state = temp
            toggle_fan()

def toggle_fan():
    if fan_state:
        fan.on()
    else:
        fan.off()

if __name__ == "__main__":
    m = load_model()
    while True:
        try:
            update(m)
        except:
            fan.off()
            break