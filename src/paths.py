import os
import os.path as path

ROOT = path.join(path.dirname(path.abspath(__file__)), os.pardir)
IMAGES = path.join(ROOT, "images")
MODEL = path.join(ROOT, "model")
TMP = path.join(ROOT, "tmp")
SRC = path.join(ROOT, "src")