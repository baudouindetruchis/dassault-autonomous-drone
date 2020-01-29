import numpy as np
import random
import os
from datetime import datetime
import time
import math
from scipy.ndimage import interpolation
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== REQUIREMENTS ==========
# input model size = 200x200px
# background size = 720x480px
#
# update the variable : path_folder
# models in folder : models/
# backgrounds in folder : backgrounds/
# outputs in folder : generated/
# ==================================

def find_coeffs(pa, pb):
    """Get coefficients for perspective transformation"""
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])
    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)
    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)

    return np.array(res).reshape(8)

def get_factor():
    """Get a random factor for image perspective transformation"""
    factor = random.randint(0,20)/100                                           # percentage of deformation

    return factor


def random_transform(model, background, label_id):
    """Pipeline that adds a model on top of a background after random transformations + return label"""
    # Random 2D rotation
    model = model.rotate(random.randint(0,359), expand=True)
    model = model.crop(model.getbbox())

    # Random 3D transform
    model_width, model_height = model.size
    coeffs = find_coeffs([(0 + get_factor()*model_width, 0 + get_factor()*model_height),                            # new upper-left corner
                          (model_width - get_factor()*model_width, 0 + get_factor()*model_height),                  # new upper-right corner
                          (model_width - get_factor()*model_width, model_height - get_factor()*model_height),       # new bottom-right corner
                          (0 + get_factor()*model_width, model_height - get_factor()*model_height)],                # new bottom-left corner
                         [(0, 0), (model_width, 0), (model_width, model_height), (0, model_height)])                # previous image shape
    model = model.transform((model_width, model_height), Image.PERSPECTIVE, coeffs)
    model = model.crop(model.getbbox())

    # Random scaling
    model_width, model_height = model.size
    scale = random.randint(20,100)/100
    model.thumbnail((model_width*scale, model_height*scale), Image.ANTIALIAS)

    # Random brithness
    enhancer = ImageEnhance.Brightness(model)
    model = enhancer.enhance(random.randint(50,100)/100)

    # Random saturation
    enhancer = ImageEnhance.Color(model)
    model = enhancer.enhance(random.randint(50,100)/100)

    # Random gaussian_noise
    alpha = model.split()[-1]
    model = model.convert('RGB')
    model_array = cv2.cvtColor(np.array(model), cv2.COLOR_RGB2BGR)              # Convert to array and swap RGB --> BGR
    noise = np.random.normal(loc=0, scale=1, size=model_array.shape)
    factor = random.randint(10,70)
    model_array = np.clip((model_array + noise*factor),0,255).astype('uint8')
    model = Image.fromarray(cv2.cvtColor(model_array, cv2.COLOR_BGR2RGB))       # Swap BGR --> RGB and convert to pillow
    model.putalpha(alpha)

    # Blur
    model = model.filter(ImageFilter.GaussianBlur(radius = 1))

    # Random paste : model --> background
    model_width, model_height = model.size
    background_width, background_height = background.size
    random_x = random.randint(0,background_width-model_width)
    random_y = random.randint(0,background_height-model_height)
    background.paste(model, (random_x, random_y))

    # Save bounding box [Label_ID, X_CENTER, Y_CENTER, WIDTH, HEIGHT]
    label = [label_id,
            (random_x + 0.5*model_width)/background_width,
            (random_y + 0.5*model_height)/background_height,
            model_width/background_width,
            model_height/background_height]

    return background, label

def random_generate(path_folder):
    """Pick randomly one model, one background and create a synthetic training example"""
    # Pick a model name & background name
    models_list = os.listdir(path_folder + 'models/')
    backgrounds_list = os.listdir(path_folder + 'backgrounds/')
    model_name = random.choice(models_list)
    background_name = random.choice(backgrounds_list)

    # Import
    model = Image.open(path_folder + 'models/' + model_name)
    background = Image.open(path_folder + 'backgrounds/' + background_name)

    # Get label_id from model name
    label_id = model_name.split('_')[0]

    # Random transform + label
    background, label = random_transform(model, background, label_id)

    return background, label


# ========== RUN ==========

path_folder = 'D:/code#/[large_data]/dassault/'

for i in range(3):
    # Get one generated image + label
    generated, label = random_generate(path_folder)
    print(label)

    # Save generated image
    timestamp = str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
    generated.save(path_folder + 'generated/' + str(label[0]) + '_' + timestamp + '.jpg')
