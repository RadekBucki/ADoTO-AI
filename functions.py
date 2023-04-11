import rasterio.features
import requests
import base64
import cv2
import json
import numpy as np
from typing import List, Dict

from InputDto import OnMapObject, Label, Picture


# compute what label each pixel of the picture should have
# default pixel label is Label.OTHER
def compute_labels_for_pixels(picture_height: int, picture_width: int, objects: Dict[Label, List[OnMapObject]]) -> \
        List[List[Label]]:
    list_of_tuples = [(item[0].value, item[1]) for item in objects.items()]
    img = rasterio.features.rasterize([list_of_tuples],
                                      out_shape=(picture_height, picture_width),
                                      default_value=Label.OTHER.value)
    # plt.imshow(img)
    picture_2d_array_classified = img.astype(int)
    labels: List[List[Label]] = [[Label(num) for num in range(len(row))] for row in picture_2d_array_classified]
    return labels


# take Picture as an input, compute OnMapObjects (vectors) from Picture's pixels (raster)
def compute_shapes_from_pixels(picture: Picture) -> List[OnMapObject]:
    # TODO implement
    return []


def request_image(height: [[str]], width: [[str]], minx: [[str]], miny: [[str]], maxx: [[str]], maxy: [[str]],
                  img_name: [[str]]):
    params = {'height': height, 'width': width, 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
    response = json.loads(requests.get('http://localhost:8080/satellite', params=params).content)
    image_array = np.frombuffer(base64.b64decode(response['base64']), dtype=np.uint8)
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    cv2.imwrite(img_name, image)
