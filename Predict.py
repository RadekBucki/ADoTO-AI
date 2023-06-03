import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
from keras.models import load_model
import Func as f
import uuid
import base64
import io

try:
    config_path = Path('./config.file')
    load_dotenv(dotenv_path=config_path)
except Exception as error:
    exit()


#+------------------------------------/
#|            VARIABLES              /
#+----------------------------------/

SIZE = int(os.environ.get("TRAIN_IMAGE_SIZE"))
filename = uuid.uuid4()
object_vectors = []

#+------------------------------------/
#|            LOAD IMAGE             /
#+----------------------------------/

images = np.zeros(shape=(1, SIZE, SIZE, 3))

#received_image = "Tutaj trzeba wrzucić zwrotke z backendu i przekonwertować obrazek"
#bytes_image =  base64.b64decode(received_image)
#img = np.asarray(Image.open(io.BytesIO(image_bytes))).astype('float')/255.

#Do lokalnego testowania
received_image = "img001.png"
img = np.asarray(Image.open(received_image)).astype('float')/255.

img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
images[0] = img

#+------------------------------------/
#|            LOAD MODEL             /
#+----------------------------------/

#loaded_model = "Tutaj zwrotka z backendu też powinna mówić jaki model należy użyć"

#Do lokalnego testowania
loaded_model = load_model(f'Datasets/dataset_house.h5')
prediction = loaded_model.predict(images)

#+------------------------------------/
#|          GET PREDICTION           /
#+----------------------------------/

predicted_mask = prediction.reshape(SIZE, SIZE)
threshold_mask = f.mask_threshold(predicted_mask, threshold=0.4)
color = threshold_mask * 255
color = color.astype(np.uint8)

#+------------------------------------/
#|          SAVE & LOAD MASK         /
#+----------------------------------/

image = Image.fromarray(color)
image.save(f'{filename}.png')
mask = cv.imread(f'{filename}.png', 0)

#+------------------------------------/
#|       GET VECTORS FROM MASK       /
#+----------------------------------/

contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

for i, contour in enumerate(contours):
    object_points = []
    z = 0
    for point in contour:
        if z % int(os.environ.get("POINT_ACC")) == 0:
            x, y = point[0]
            object_points.append([x, y])
        z = z + 1

    object_vectors.append(object_points)

#Do lokalnego testowania
print(object_vectors)

#+------------------------------------/
#|           REMOVE PHOTO            /
#+----------------------------------/

f.delete_image(filename)