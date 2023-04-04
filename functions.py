from InputDto import Label, OnMapObject
from shapely.geometry import Polygon
import requests
import base64
import cv2
import json
import numpy as np


# compute what label each picture should have, based on the labeled shapes
def compute_labels_for_pixels(picture: [[int]], objects: [OnMapObject]) -> [[Label]]:
    rows: int = len(picture)
    columns: int = len(picture[0])
    # assign label to each pixel
    return [(Label.OTHER for i in range(columns)) for j in range(rows)]


# compute labeled shapes based on each pixel's label
def compute_shapes_from_pixels(picture: [[int]], objects: [OnMapObject]) -> [OnMapObject]:
    # extract objects from labeled pixels
    return [OnMapObject(Polygon([[0, 0], [1, 0], [1, 1], [0, 1]]), Label.HOUSE) for _ in range(10)]


def request_image(height: [[str]], width: [[str]],  minx: [[str]],  miny: [[str]],  maxx: [[str]],  maxy: [[str]], img_name: [[str]]):
    params = {'height': height, 'width': width, 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
    response = json.loads(requests.get('http://localhost:8080/satellite', params=params).content)
    image_array = np.frombuffer(base64.b64decode(response['base64']), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite(img_name, image) 