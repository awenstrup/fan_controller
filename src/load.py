"""Given an S3 bucket with some data, load it to the local machine.

If data is labeled (assume catagorically), sort it by directory.

If not, download to the unlabeled directory
"""
# Imports

# Base Python
import time
import datetime
import json
import logging
import os
from os import path

# Exptended Python
import cv2
import boto3

# Globals
BUCKET = "training-set-0000"
PREFIX = "tito/images"
MANIFEST = "tito/labeled/training-dataset-0000/manifests/output/output.manifest"
LABEL = "tito-or-not"
LABEL_MAP = {0: "tito", 1: "not_tito"}  # Map from labels to directory names
FILE_TYPE = ".png"

# Create Logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def load_one(key, dest, conn):
    """Download a single image

    Args:
        key (str): The object key of the file to download
        dest (str): The place to download it, preferably an absolute path
        conn: An S3 client object
    """
    file_name = dest + '/' + path.basename(key)
    with open(file_name, 'wb') as data:
        conn.download_fileobj(BUCKET, key, data)

def load_all(destination: str, manifest: str = ""):
    """Load all files to the specified destiniation

    Args:
        destiniation (str): The directory to load files to
        manifest (str): The manifest file to use, if data is already labeled
    """
    s3_client = boto3.client('s3')
    response = s3_client.list_objects(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=2000)
    objects = response["Contents"]

    if manifest:
        logger.debug("Data is already labeled, using a manifest file...")
        if not path.exists(path.join(paths.TMP, "manifest"):
            logger.debug("Downloading manifest file...")
            os.mkdir(paths.TMP)
            s3_client.download_file(BUCKET, manifest, path.join(paths.TMP, "manifest"))

        with open(path.join(paths.TMP, "manifest"), "r") as f:
            lines = f.readlines()
        lookup = {}
        for line in lines:
            d = json.loads(line)
            key = d["source-ref"].replace(f"s3://{BUCKET}/", "")
            lookup[key] = LABEL_MAP[d[LABEL]]
        
    if not path.exists(destination):
        os.mkdir(destination)

    logger.info(f"Downloading {len(objects)} files now")
    count = 0
    for obj in objects:
        key = obj["Key"]
        if FILE_TYPE in key:  # Ignore extra files, like manifests or json
            dest = path.abspath(destination + "/" + lookup.get(obj["Key"], "unlabeled"))
            if not path.exists(dest):
                os.mkdir(dest)
            load_one(key, dest, s3_client)

        count += 1
        if count % 10 == 0:
            logger.info(f"Downloaded {count} files, {len(objects) - count} to go...")


if __name__ == "__main__":
    load_all(paths.IMAGES, MANIFEST)
    os.rmdir(paths.TMP)