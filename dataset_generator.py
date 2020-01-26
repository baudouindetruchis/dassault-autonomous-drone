from PIL import Image
from random import randint

# ========== INFORMATION ==========
# --> models smaller than 300x300px
# =================================

def random_image(model, background):
    # Random rotation
    model = model.rotate(randint(0,359), expand=True)

    # Random scaling
    model_width, model_height = model.size
    scale = randint(10,100)/100
    model.thumbnail((model_width*scale, model_height*scale), Image.ANTIALIAS)

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
generated.show()
