import numpy as np
import random
import os
import glob
from datetime import datetime
import time
import math
import sys
from tqdm import tqdm
from scipy.ndimage import interpolation
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== REQUIREMENTS ==========
# input model size = ~300x300px
# background size = larger than 720x480px
#
# update the variable : path_folder
# models in folder : models/
# fake models in folder : models_fake/
# backgrounds in folder : backgrounds/
# outputs in folder : generated/
# ==================================

# ========== INFORMATION ===========
# ~10 images/sec
# higher altitude --> maybe crop before blob
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

def random_transform(model, background, label_id, max_scale=100):
    """Pipeline that adds a model on top of a background after random transformations + return label"""
    # Random 2D rotation
    model = model.rotate(random.randint(0,359), expand=True)
    model = model.crop(model.getbbox())

    # Random 3D transform
    model_width, model_height = model.size
    width_factor = random.randint(0,20)/100
    height_factor = random.randint(0,20)/100
    coeffs = find_coeffs([(0 + width_factor*model_width, 0 + height_factor*model_height),                   # new upper-left corner
                          (model_width - width_factor*model_width, 0 + height_factor*model_height),         # new upper-right corner
                          (model_width, model_height),                                                      # new bottom-right corner
                          (0, model_height)],                                                               # new bottom-left corner
                         [(0, 0), (model_width, 0), (model_width, model_height), (0, model_height)])        # previous image shape
    model = model.transform((model_width, model_height), Image.PERSPECTIVE, coeffs)
    model = model.crop(model.getbbox())

    # Second 2D rotation
    model = model.rotate(random.randint(0,359), expand=True)
    model = model.crop(model.getbbox())

    # Random scaling
    model_width, model_height = model.size
    scale = random.randint(15,max_scale)/100
    model.thumbnail((model_width*scale, model_height*scale), Image.ANTIALIAS)

    # Random brithness
    enhancer = ImageEnhance.Brightness(model)
    model = enhancer.enhance(random.randint(50,100)/100)

    # Random saturation
    enhancer = ImageEnhance.Color(model)
    model = enhancer.enhance(random.randint(50,100)/100)

    # Blur
    model = model.filter(ImageFilter.GaussianBlur(radius=random.randint(0,2)))  # Leave extra room for noise generation

    # Random gaussian_noise
    alpha = model.split()[-1]
    model = model.convert('RGB')
    model_array = cv2.cvtColor(np.array(model), cv2.COLOR_RGB2BGR)              # Convert to array and swap RGB --> BGR
    noise = np.random.normal(loc=0, scale=1, size=model_array.shape)
    factor = random.randint(10,250)                                             # Noise intensity
    model_array = np.clip((model_array + noise*factor),0,255).astype('uint8')
    model = Image.fromarray(cv2.cvtColor(model_array, cv2.COLOR_BGR2RGB))       # Swap BGR --> RGB and convert to pillow
    model.putalpha(alpha)

    # Random blur
    model = model.filter(ImageFilter.GaussianBlur(radius=random.randint(1,2)))

    # Rolling effect
    model_array = np.array(model)
    amplitude = random.randint(0,5)
    period = random.randint(5,40)/100
    shift = lambda x: amplitude * np.sin(2.0*np.pi*x*period)
    orientation = random.randint(0,1)
    if orientation == 1:
        for i in range(model_array.shape[1]):
            model_array[:,i] = np.roll(model_array[:,i], int(shift(i)), axis=0) # axis=0 so the matrix is not flattened
    else:
        for i in range(model_array.shape[0]):
            model_array[i,:] = np.roll(model_array[i,:], int(shift(i)), axis=0)
    model = Image.fromarray(model_array)

    # Random paste : model --> background
    model_width, model_height = model.size
    background_width, background_height = background.size
    random_x = random.randint(0,background_width-model_width)
    random_y = random.randint(0,background_height-model_height)
    background.paste(model, (random_x, random_y), model)                        # Add model's alpha as a mask

    # Save bounding box [Label_ID, X_CENTER, Y_CENTER, WIDTH, HEIGHT]
    label = [label_id,
            (random_x + 0.5*model_width)/background_width,
            (random_y + 0.5*model_height)/background_height,
            model_width/background_width,
            model_height/background_height]

    return background, label

def background_generate(path_folder, size):
    """Generate a large amount of backgrounds with correct size"""
    # List source backgrounds
    backgrounds_list = os.listdir(path_folder + 'backgrounds/')
    backgrounds_list = [i for i in backgrounds_list if 'backgen_' not in i]

    for i in tqdm(range(size), desc='Generating backgrounds'):
        # Randomly choose a source background
        background_name = random.choice(backgrounds_list)
        background = Image.open(path_folder + 'backgrounds/' + background_name)

        # Crop background
        width, height = background.size
        random_x = random.randint(0, width - 720)
        random_y = random.randint(0, height - 480)
        generated = background.crop((random_x, random_y, random_x+720, random_y+480))

        # Random transform
        enhancer = ImageEnhance.Brightness(generated)
        generated = enhancer.enhance(random.randint(70,130)/100)
        enhancer = ImageEnhance.Color(generated)
        generated = enhancer.enhance(random.randint(70,100)/100)

        # Save generated background
        timestamp = str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
        generated.save(path_folder + 'backgrounds/' + 'backgen_' + timestamp + '.jpg')

def random_generate(path_folder):
    """Create a synthetic training example"""
    # Pick model & background names
    models_list = os.listdir(path_folder + 'models/')
    models_fake_list = os.listdir(path_folder + 'models_fake/')
    backgrounds_list = os.listdir(path_folder + 'backgrounds/')
    backgrounds_list = [i for i in backgrounds_list if 'backgen_' in i]

    model_name = random.choice(models_list)
    model_fake_names = random.choices(models_fake_list, k=random.randint(0,2))
    background_name = random.choice(backgrounds_list)

    # Load model & background
    model = Image.open(path_folder + 'models/' + model_name)
    background = Image.open(path_folder + 'backgrounds/' + background_name)

    # Add fake models
    for fake_name in model_fake_names:
        fake = Image.open(path_folder + 'models_fake/' + fake_name)
        background, _ = random_transform(fake, background, 'x', max_scale=70)

    # Add model + get label
    label_id = model_name.split('_')[0]
    background, label = random_transform(model, background, label_id)

    return background, label


# ========== RUN ==========

path_folder = 'D:/code#/[large_data]/dassault/'
# path_folder = '/media/bdn/Data/code#/[large_data]/dassault/'

# Get number of synthetic images needed
size = int(sys.argv[1])

# Generate random backgrounds
background_generate(path_folder, size//5)

# Generate synthetic images
for i in tqdm(range(size), desc='Generating synthetic images'):
    # Get one generated image + label
    generated, label = random_generate(path_folder)

    # Save generated image
    filename = str(label[0]) + '_' + str(round(datetime.utcnow().timestamp())) + '_' + str(round(datetime.utcnow().timestamp()*1000))[-3:]
    generated.save(path_folder + 'generated/' + filename + '.jpg')

    # Save corresponding label
    with open(path_folder + 'generated_labels/' + filename + '.txt', 'w+') as file:
        for count, chunk in enumerate(label):
            if count == 4:
                file.write(str(chunk))
            else:
                file.write(str(chunk) + ' ')

# Remove generated backgrounds
backgrounds_list = os.listdir(path_folder + 'backgrounds/')
backgrounds_list = [i for i in backgrounds_list if 'backgen_' in i]
for i in tqdm(backgrounds_list, desc='Removing generated backgrounds'):
    os.remove(path_folder + 'backgrounds/' + i)

# Train test split file
percentage_test = 10
index_test = round(100/percentage_test)

file_train = open(path_folder + 'train.txt', 'w')
file_test = open(path_folder + 'test.txt', 'w')

for count, item in enumerate(os.listdir(path_folder + 'generated/')):
    if count % index_test == 0:
        file_test.write('data/images/' + item + "\n")
    else:
        file_train.write('data/images/' + item + "\n")
