import validate_img
import pandas as pd
import numpy as np
import os

FACE_DIR = "images/face/"

# Call the validate_image_format function to check if the image format is correct
validate_img.validate_image_format(FACE_DIR)