import os
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
import cv2 as cv
from PIL import Image
from keras.models import load_model
import Func as f
import uuid
import json
import base64
import io


def find_shapes(img, object_type, width):
    try:
        config_path = Path('./config.file')
        load_dotenv(dotenv_path=config_path)
    except Exception as error:
        exit()

    # +------------------------------------/
    # |            VARIABLES              /
    # +----------------------------------/

    SIZE = int(os.environ.get(object_type.swapcase() + "_TRAIN_IMAGE_SIZE"))
    filename = uuid.uuid4()
    object_vectors = []

    # +------------------------------------/
    # |            LOAD IMAGE             /
    # +----------------------------------/

    images = np.zeros(shape=(1, SIZE, SIZE, 3))
    received_image = img
    bytes_image = base64.b64decode(received_image)
    img = np.asarray(Image.open(io.BytesIO(bytes_image))).astype('float') / 255.
    img = cv.resize(img, (SIZE, SIZE), cv.INTER_AREA)
    images[0] = img

    # +------------------------------------/
    # |            LOAD MODEL             /
    # +----------------------------------/

    loaded_model = load_model(f'Datasets/dataset_' + object_type + '.h5')
    prediction = loaded_model.predict(images)

    # +------------------------------------/
    # |          GET PREDICTION           /
    # +----------------------------------/

    predicted_mask = prediction.reshape(SIZE, SIZE)
    threshold_mask = f.mask_threshold(predicted_mask, threshold=0.4)
    color = threshold_mask * 255
    color = color.astype(np.uint8)

    # +------------------------------------/
    # |          SAVE & LOAD MASK         /
    # +----------------------------------/

    image = Image.fromarray(color)
    image.save(f'{filename}.png')
    mask = cv.imread(f'{filename}.png', 0)

    # +------------------------------------/
    # |       GET VECTORS FROM MASK       /
    # +----------------------------------/

    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        object_points = []
        z = 0
        for point in contour:
            single_point = {}
            if z % int(os.environ.get("POINT_ACC")) == 0:
                single_point.update({"x": float(point[0][0] * float(width)/SIZE),
                                     "y": float(point[0][1] * float(width)/SIZE)})
                object_points.append(single_point)
            z = z + 1
        object_points.append(object_points[0])
        if len(object_points) >= 3:
            object_vectors.append(object_points)

    # +------------------------------------/
    # |           REMOVE PHOTO            /
    # +----------------------------------/
    f.delete_image(filename)
    return object_vectors


def create_response(img, model, width):
    return json.dumps(find_shapes(img, model, width), indent=2)
