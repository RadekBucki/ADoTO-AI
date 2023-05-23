import rasterio.features
import requests
import json
from Predict import create_response
from typing import List, Dict
from shapely.geometry import shape

from InputDto import OnMapObject, Label, Picture


# compute what label each pixel of the picture should have
# default pixel label is Label.OTHER
def compute_labels_for_pixels(picture_height: int, picture_width: int, objects: Dict[Label, List[OnMapObject]]) -> \
        List[List[Label]]:
    list_of_tuples = [(item[0].value, item[1]) for item in objects.items()]
    img = rasterio.features.rasterize([list_of_tuples],
                                      out_shape=(picture_height, picture_width),
                                      default_value=Label.OTHER.value)
    picture_2d_array_classified = img.astype(int)
    labels: List[List[Label]] = [[Label(num) for num in range(len(row))] for row in picture_2d_array_classified]
    return labels


# take Picture as an input, compute OnMapObjects (vectors) from Picture's pixels (raster)
def compute_shapes_from_pixels(picture: Picture) -> List[OnMapObject]:
    polygons: List[OnMapObject] = []
    for vec, category_index in rasterio.features.shapes(picture.array):
        label = Label(category_index)
        if label.is_meaningful():
            polygons.append(OnMapObject(shape(vec), label))
    return polygons


def request_image(width: [[str]], minx: [[str]], miny: [[str]], maxx: [[str]], maxy: [[str]],
                  model: [[str]]):
    params = {'width': width, 'minx': minx, 'miny': miny, 'maxx': maxx, 'maxy': maxy}
    # Note: Backend should send the picture automatically
    response = json.loads(requests.get('http://localhost:8080/geoportal/satellite/epsg2180', params=params).content)
    return create_response(response['base64'], model)
