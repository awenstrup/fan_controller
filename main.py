# Imports

# Base Python
import time
import datetime

# Exptended Python
import cv2
import boto3

def take_picture():
    """Take a picture and save it"""
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    return_value, image = camera.read()
    del(camera)

    time_str = str(datetime.datetime.now()).replace(" ", "_").replace("-", "_").replace(":", "_")
    file_name = f"images/{time_str}.png"
    cv2.imwrite(file_name, image)
    return file_name

def upload_s3(file_name: str):
    """Upload an image to S3"""
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(file_name, "awenstrup", f"tito/{file_name}")
    except ClientError as e:
        logging.error(e)
        return False
    return True

while True:
    file_name = take_picture()
    upload_s3(file_name)
    time.sleep(10 * 60)