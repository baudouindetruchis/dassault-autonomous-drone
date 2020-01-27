import numpy as np
from random import randint
from PIL import Image, ImageEnhance, ImageFilter
import cv2

# ========== INFORMATION ==========
# input model size = 200x200px
# background size = 720x480px
# =================================

def random_image(model, background):
    # Random rotation
    model = model.rotate(randint(0,359), expand=True)
    model = model.crop(model.getbbox())

    # Random scaling
    model_width, model_height = model.size
    scale = randint(10,100)/100
    model.thumbnail((model_width*scale, model_height*scale), Image.ANTIALIAS)

    # Random overall brithness
    enhancer = ImageEnhance.Brightness(model)
    model = enhancer.enhance(randint(50,100)/100)

    # Random overall saturation
    enhancer = ImageEnhance.Color(model)
    model = enhancer.enhance(randint(50,100)/100)

    # Random gaussian_noise
    alpha = model.split()[-1]
    model = model.convert('RGB')
    model_array = cv2.cvtColor(np.array(model), cv2.COLOR_RGB2BGR)              # Convert to array and swap RGB --> BGR
    noise = np.random.normal(loc=0, scale=1, size=model_array.shape)
    factor = randint(10,70)
    model_array = np.clip((model_array + noise*factor),0,255).astype('uint8')
    model = Image.fromarray(cv2.cvtColor(model_array, cv2.COLOR_BGR2RGB))       # Swap BGR --> RGB and convert to pillow
    model.putalpha(alpha)

    # Blur
    model = model.filter(ImageFilter.GaussianBlur(radius = 1))

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
