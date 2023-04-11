from shapely.geometry import Polygon
from enum import Enum
from typing import List, Dict

import functions


# represents all classes on the map that we are interested in
class Label(Enum):
    # meaningful
    HOUSE = 1
    FIELD = 2
    WATER_BODY = 3
    # meaningless
    OTHER = 4

    # returns true if label value represents a meaningful object on the map
    def is_meaningful(self):
        return self != Label.OTHER


class Picture:
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        # TODO array should have real pixel data here
        self.array: List[List[int]] = []


# represents object on the map, shape & its label
class OnMapObject:
    def __init__(self, polygon: Polygon, label: Label):
        self.polygon = polygon
        if not label.is_meaningful():
            raise Exception(f"Object on the map should be meaningful, {label} does not count")
        self.label = label


# represents satellite picture with all pixels having a number representing class of object
class InputDto:
    def __init__(self, picture: Picture, objects: Dict[Label, List[OnMapObject]]):
        self.picture: Picture = picture
        self.categories: List[List[Label]] = functions.compute_labels_for_pixels(picture.height, picture.width, objects)
