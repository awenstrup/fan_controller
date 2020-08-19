"""Script to be run on Raspberry Pi
Periodically takes an image (with attached camera), 
and uploads it to an S3 bucket.
"""
# Imports

# Exptended Python
import cv2
import boto3

# Globals
PERIOD = 10  # 10 seconds

def update():
    camera = cv2.VideoCapture(0)
    time.sleep(1)
    _, image = camera.read()
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    del(camera)

    

def upload_s3(file_name: str):
    """Upload an image to S3
    
    Args:
        file_name (str): The relative path to the image to upload
        
    Returns:
        bool: Whether or not the upload succeeded
    """
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
    time.sleep(PERIOD)