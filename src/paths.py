import os
import os.path as path

ROOT = path.abspath(path.join(path.dirname(path.abspath(__file__)), os.pardir))
IMAGES = path.join(ROOT, "images")
MODEL = path.join(ROOT, "model")
LITE_MODEL = path.join(ROOT, "lite_model")
TMP = path.join(ROOT, "tmp")
SRC = path.join(ROOT, "src")
PI = path.join(SRC, "pi")
