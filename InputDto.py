from shapely.geometry import Polygon
from enum import Enum
from typing import List, Dict

import functions


# represents all classes on the map that we are interested in
class Label(Enum):
    HOUSE = 1
    FIELD = 2
    WATER_BODY = 3
    OTHER = 4


class Picture:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height


# represents object on the map, shape & its label
class OnMapObject:
    def __init__(self, polygon: Polygon, label: Label):
        self.polygon = polygon
        self.label = label


# represents satellite picture with all pixels having a number representing class of object
class InputDto:
    def __init__(self, picture: Picture, objects: Dict[Label, List[OnMapObject]]):
        self.picture: Picture = picture
        self.categories: List[List[Label]] = functions.compute_labels_for_pixels(picture.height, picture.width, objects)
