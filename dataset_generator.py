import numpy as np
from random import randint
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== INFORMATION ==========
# input model size = 200x200px
# =================================

def random_image(model, background):
    # Random rotation
    model = model.rotate(randint(0,359), expand=True)

    # Random scaling
    model_width, model_height = model.size
    scale = randint(10,100)/100
    model.thumbnail((model_width*scale, model_height*scale), Image.ANTIALIAS)

    # Random brithness
    enhancer = ImageEnhance.Brightness(model)
    model = enhancer.enhance(randint(50,100)/100)

    # Random gaussian_noise
    alpha = model.split()[-1]
    model = model.convert('RGB')
    model_array = cv2.cvtColor(np.array(model), cv2.COLOR_RGB2BGR)              # Convert to array and swap RGB --> BGR
    print(model_array.max())
    noise = np.random.normal(loc=0, scale=1, size=model_array.shape)
    model_array = np.clip((model_array + noise).astype('uint8'),0,255)
    model = Image.fromarray(cv2.cvtColor(model_array, cv2.COLOR_BGR2RGB))       # Swap BGR --> RGB and convert to pillow
    model.putalpha(alpha)

    # Random paste : model --> background
    model_width, model_height = model.size
    background_width, background_height = background.size
    background.paste(model, (randint(0,background_width-model_width),randint(0,background_height-model_height)), model)

    return background

# =================================

path_data = 'D:/code#/[large_data]/dassault/'

image_model = Image.open(path_data + 'models/redarrow_model1.png')
image_background = Image.open(path_data + 'backgrounds_real/testflight_2019-12-14_14-36-13.jpg')

generated = random_image(image_model, image_background)
generated = random_image(image_model, generated)
generated = random_image(image_model, generated)
generated = random_image(image_model, generated)
generated = random_image(image_model, generated)
generated = random_image(image_model, generated)
generated = random_image(image_model, generated)

generated.show()

# model_array = cv2.cvtColor(np.array(image_model), cv2.COLOR_RGB2BGR)
# print(model_array.min())
