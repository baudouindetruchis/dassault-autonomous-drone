import numpy as np
import random
import os
from datetime import datetime
import time
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== REQUIREMENTS ==========
# background input >> 720x480px
# ==================================

def background_transform(background):
    enhancer = ImageEnhance.Brightness(background)
    background = enhancer.enhance(random.randint(70,130)/100)
    enhancer = ImageEnhance.Color(background)
    background = enhancer.enhance(random.randint(70,100)/100)

    return background


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
        regular = background_transform(generated)
        timestamp = str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
        regular.save(path_folder + 'backgrounds_internet/' + 'internet_' + timestamp + '.jpg')

        # Save mirror transform LEFT_RIGHT
        flipped = generated.transpose(Image.FLIP_LEFT_RIGHT)
        flipped = background_transform(flipped)
        timestamp = str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
        flipped.save(path_folder + 'backgrounds_internet/' + 'internet_' + timestamp + '.jpg')
