"""Script to be run on Raspberry Pi
Periodically takes an image (with attached camera), 
and uploads it to an S3 bucket.

TODO: Add IAM role for this, don't use root permissions!
"""
# Imports

# Base Python
import time
import datetime

# Exptended Python
import cv2
import boto3

# Globals
PERIOD = 2 * 60  # 2 minutes


def take_picture() -> str:
    """Take a picture and save it
    
    Returns:
        str: The relative path to the image
    """
    camera = cv2.VideoCapture(0)
    time.sleep(1)  # Wait for the camera to initialize
    return_value, image = camera.read()
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    del camera

    time_str = (
        str(datetime.datetime.now())
        .replace(" ", "_")
        .replace("-", "_")
        .replace(":", "_")
    )
    file_name = f"images/{time_str}.png"
    cv2.imwrite(file_name, image)
    return file_name


def upload_s3(file_name: str):
    """Upload an image to S3
    
    Args:
        file_name (str): The relative path to the image to upload
        
    Returns:
        bool: Whether or not the upload succeeded
    """
    s3_client = boto3.client("s3")
    try:
        response = s3_client.upload_file(file_name, "awenstrup", f"tito/{file_name}")
    except ClientError as e:
        logging.error(e)
        return False
    return True


while True:
    file_name = take_picture()
    upload_s3(file_name)
    time.sleep(PERIOD)
