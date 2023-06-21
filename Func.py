import matplotlib.pyplot as plt
import numpy as np
from dotenv import load_dotenv
from pathlib import Path
from PIL import Image

import os

try:
    config_path = Path('./config.file')
    load_dotenv(dotenv_path=config_path)
except Exception as error:
    exit()

SIZE = int(os.environ.get("TRAIN_IMAGE_SIZE"))

def verify_images(images, masks):
    plt.figure(figsize=(10, 15))
    for i in range(1, 11):
        plt.subplot(5, 2, i)

    id = np.random.randint(len(images))
    if i % 2 != 0:
        plt.imshow(images[id], cmap=None)
        plt.title('Original Image')
    elif i % 2 == 0:
        plt.imshow(masks[id].reshape(SIZE, SIZE), cmap='gray')
        plt.title('Mask')

    plt.tight_layout()
    plt.show()

def mask_threshold(image, threshold=0.25):
  return image > threshold

def prediction_results(images_test, mask_test, predictions, threshold):
    for i in range(5):
        plt.figure(figsize=(10,5))
        k = np.random.randint(len(images_test))
        original_image = images_test[k]
        mask = mask_test[k].reshape(SIZE,SIZE)
        predicted_mask = predictions[k].reshape(SIZE,SIZE)
        threshold_mask = mask_threshold(predicted_mask, threshold=threshold)

        plt.figure(figsize=(15,5))

        plt.subplot(1,4,1)
        plt.imshow(original_image);plt.title('Orginal Image')

        plt.subplot(1,4,2)
        plt.imshow(mask, cmap='gray');plt.title('Original Mask')

        plt.subplot(1,4,3)
        plt.imshow(predicted_mask, cmap='gray');plt.title('Predicted Mask')
        
        plt.subplot(1,4,4)
        plt.imshow(threshold_mask, cmap='gray');plt.title(f'Predicted Mask with cutoff={threshold}')

        plt.tight_layout()
        plt.show()

def delete_image(filename):
    file = f'{filename}.png'
    if os.path.exists(file):
        os.remove(file)

def crop_photos_and_save(image_path, output_directory, crop_size, filename):
    image = Image.open(image_path)
    
    width, height = image.size
    
    num_cols = width // crop_size
    num_rows = height // crop_size
    
    for col in range(num_cols):
        for row in range(num_rows):
            left = col * crop_size
            upper = row * crop_size
            right = left + crop_size
            lower = upper + crop_size
            
            cropped_image = image.crop((left, upper, right, lower))
            output_path = f"{output_directory}/{filename}_{col}_{row}.jpg"
            
            if cropped_image.mode == "RGBA":
                cropped_image = cropped_image.convert("RGB")

            cropped_image.save(output_path, "JPEG")