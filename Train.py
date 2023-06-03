import os
from dotenv import load_dotenv
from pathlib import Path

try:
    config_path = Path('./config.file')
    load_dotenv(dotenv_path=config_path)
except Exception as error:
    exit()

os.add_dll_directory(os.environ.get("CUDA_TOOLKIT_BIN"))
os.add_dll_directory(os.environ.get("CUDA_TOOLKIT_LIB"))
os.add_dll_directory(os.environ.get("CUDA_TOOLKIT_ZLIBWAPI"))

import numpy as np
from tqdm import tqdm
import cv2 as cv
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import Model
from keras.layers import Input, Conv2D, Conv2DTranspose, MaxPooling2D, concatenate, Dropout
import tensorflow as tf
import Func as f

#+------------------------------------/
#|            VARIABLES              /
#+----------------------------------/
SIZE = int(os.environ.get("TRAIN_IMAGE_SIZE"))

#+------------------------------------/
#|                UNET               /
#+----------------------------------/

def unet_model(input_layer, start_neurons):
  # Contraction path
  conv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(input_layer)
  conv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(conv1)
  pool1 = MaxPooling2D((2, 2))(conv1)
  pool1 = Dropout(0.25)(pool1)

  conv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(pool1)
  conv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(conv2)
  pool2 = MaxPooling2D((2, 2))(conv2)
  pool2 = Dropout(0.5)(pool2)

  conv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(pool2)
  conv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(conv3)
  pool3 = MaxPooling2D((2, 2))(conv3)
  pool3 = Dropout(0.5)(pool3)

  conv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(pool3)
  conv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(conv4)
  pool4 = MaxPooling2D((2, 2))(conv4)
  pool4 = Dropout(0.5)(pool4)

  # Middle
  convm = Conv2D(start_neurons*16, kernel_size=(3, 3), activation="relu", padding="same")(pool4)
  convm = Conv2D(start_neurons*16, kernel_size=(3, 3), activation="relu", padding="same")(convm)

  # Expansive path
  deconv4 = Conv2DTranspose(
      start_neurons*8, kernel_size=(3, 3), strides=(2, 2), padding="same")(convm)
  uconv4 = concatenate([deconv4, conv4])
  uconv4 = Dropout(0.5)(uconv4)
  uconv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(uconv4)
  uconv4 = Conv2D(start_neurons*8, kernel_size=(3, 3), activation="relu", padding="same")(uconv4)

  deconv3 = Conv2DTranspose(start_neurons*4, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv4)
  uconv3 = concatenate([deconv3, conv3])
  uconv3 = Dropout(0.5)(uconv3)
  uconv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(uconv3)
  uconv3 = Conv2D(start_neurons*4, kernel_size=(3, 3), activation="relu", padding="same")(uconv3)

  deconv2 = Conv2DTranspose(start_neurons*2, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv3)
  uconv2 = concatenate([deconv2, conv2])
  uconv2 = Dropout(0.5)(uconv2)
  uconv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(uconv2)
  uconv2 = Conv2D(start_neurons*2, kernel_size=(3, 3), activation="relu", padding="same")(uconv2)

  deconv1 = Conv2DTranspose(start_neurons*1, kernel_size=(3, 3), strides=(2, 2), padding="same")(uconv2)
  uconv1 = concatenate([deconv1, conv1])
  uconv1 = Dropout(0.5)(uconv1)
  uconv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(uconv1)
  uconv1 = Conv2D(start_neurons, kernel_size=(3, 3), activation="relu", padding="same")(uconv1)

  # Last conv and output
  output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(uconv1)

  return output_layer

  # Compile unet model

#+------------------------------------/
#|               MODEL               /
#+----------------------------------/

input_layer = Input((SIZE, SIZE, 3))
output_layer = unet_model(input_layer=input_layer, start_neurons=16)

model = Model(input_layer, output_layer)
model.compile(loss="binary_crossentropy",
              optimizer="adam", metrics=["accuracy"])
model.summary()

#+------------------------------------/
#|            LOAD IMAGES            /
#+----------------------------------/

image_path = f'{os.environ.get("IMAGE_PATH")}'
mask_path = f'{os.environ.get("MASK_PATH")}'

image_names = sorted(next(os.walk(image_path))[-1])
mask_names = sorted(next(os.walk(mask_path))[-1])

#+------------------------------------/
#|         CONVERT IMAGES            /
#+----------------------------------/

images = np.zeros(shape=(len(image_names), SIZE, SIZE, 3))
masks = np.zeros(shape=(len(image_names), SIZE, SIZE, 1))

for id in tqdm(range(len(image_names)), desc="Images"):
  path = image_path + image_names[id]
  img = np.asarray(Image.open(path)).astype('float')/255.
  img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
  images[id] = img

for id in tqdm(range(len(mask_names)), desc="Mask"):
  path = mask_path + mask_names[id]
  mask = np.asarray(Image.open(path).convert("L")).astype('float')/255.
  mask = cv.resize(mask, (SIZE, SIZE), cv.INTER_AREA)
  mask = np.expand_dims(mask, axis=-1)
  masks[id] = mask

#+------------------------------------/
#|          VERIFY IMAGES            /
#+----------------------------------/

f.verify_images(images, masks)

#+------------------------------------/
#|           IMAGE SPLIT             /
#+----------------------------------/

images_train, images_test, mask_train, mask_test = train_test_split(images, masks, test_size=0.25)

#+------------------------------------/
#|        TRAIN CONFIGURATION        /
#+----------------------------------/

epochs = int(os.environ.get("EPOCH_SIZE"))
batch_size = int(os.environ.get("BATCH_SIZE"))

history = model.fit(images_train, mask_train,
                    validation_data=[images_test, mask_test], 
                    epochs=epochs,
                    batch_size=batch_size)


predictions = model.predict(images_test)
  
f.prediction_results(images_test, mask_test, model.predict(images_test), 0.5)

model.save(f'{os.environ.get("MODEL_NAME")}.h5')
