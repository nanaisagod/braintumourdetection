import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# Set the path to the dataset
tumor_location = '/Users/aaravnanavaty/Downloads/archive(11)/Training'
IMG_Size = 100
loaded_model = load_model('tumor_detection.h5')

# Load the location detection model
loaded_model2 = load_model('tumor_location.h5')

image_path = '/Users/aaravnanavaty/Downloads/archive(8)/Prediction_check_images/Prediction_check_images/pituitary2.jpg'

try:
    with Image.open(image_path) as img:
        img = img.convert("L")
        img = img.resize((IMG_Size, IMG_Size))
        img.save('input_image.png', 'PNG')
except Exception as e:
    print(f"Failed to convert the image: {str(e)}")
    exit()

image = cv2.imread('input_image.png', cv2.IMREAD_GRAYSCALE)

if image is not None:
    IMG_Size = 100
    image = image / 255.0
    input_data = np.expand_dims(image, axis=0)


    predictions = loaded_model.predict(input_data)

    if predictions < 0.5:
        print("Tumor is present.")


        location_predictions = loaded_model2.predict(input_data)

        max_index = np.argmax(location_predictions)

        if max_index == 0:
            print("Type: Glioma")
        elif max_index == 1:
            print("Type: Meningioma")
        elif max_index == 2:
            print("Type: Pituitary")
        else:
            print("Unknown Type")
    else:
        print("No tumor is present.")
else:
    print(f"Failed to load the image at {image_path}")

## could i ask user to input image path and use that path instead of pre-set path?
