import numpy as np
import random
import os
from datetime import datetime
import time
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== INFORMATION ==========
# input model size = 200x200px
# background size = 720x480px
# =================================

def random_transform(model, background, label_id):
    # Random rotation
    model = model.rotate(random.randint(0,359), expand=True)
    model = model.crop(model.getbbox())

    # Random scaling
    model_width, model_height = model.size
    scale = random.randint(10,100)/100
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
    background.paste(model, (random_x, random_y), model)

    # Save bounding box [Label_ID, X_CENTER, Y_CENTER, WIDTH, HEIGHT]
    label = [label_id, random_x, random_y, random_x+model_width, random_y+model_height]

    return background, label

def random_generate(path_folder):
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


# =================================

path_folder = 'D:/code#/[large_data]/dassault/'

for i in range(100):
    # Get one generated image + label
    generated, label = random_generate(path_folder)

    # Save generated image
    timestamp = str(round(datetime.utcnow().timestamp()))
    generated.save(path_folder + 'generated/' + str(label[0]) + '_' + timestamp + '.jpg')

    time.sleep(1)
