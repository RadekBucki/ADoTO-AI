from shapely.geometry import Polygon
from enum import Enum

import functions


# represents all classes on the map that we are interested in
class Label(Enum):
    HOUSE = 1
    FIELD = 2
    WATER_BODY = 3
    OTHER = 4


# represents object on the map, shape & its label
class OnMapObject:
    def __init__(self, polygon: Polygon, label: Label):
        self.polygon = polygon
        self.label = label


# represents satellite picture with object that it contains
class InputDto:
    def __init__(self, picture: [[int]], objects: [OnMapObject]):
        self.picture: [[int]] = picture
        self.labels: [[Label]] = functions.compute_labels_for_pixels(picture=picture, objects=objects)
