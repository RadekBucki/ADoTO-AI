from InputDto import Label, OnMapObject
from shapely.geometry import Polygon


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
