import numpy as np
import random
import os
from datetime import datetime
import time
from PIL import Image, ImageEnhance, ImageFilter
import cv2


# ========== RUN ==========

path_folder = 'D:/code#/[large_data]/dassault/'

backgrounds_list = os.listdir(path_folder + 'backgrounds_temp/')

for background_name in backgrounds_list:
    background = Image.open(path_folder + 'backgrounds_temp/' + background_name)

    for i in range(10):
        # Crop background
        width, height = background.size
        random_x = random.randint(0, width - 720)
        random_y = random.randint(0, height - 480)
        generated = background.crop((random_x, random_y, random_x+720, random_y+480))

        # Save generated background
        generated
