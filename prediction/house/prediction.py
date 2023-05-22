import numpy as np
import torch
from skimage import io
from skimage.transform import resize

from models.backbone import DetectionBranch, NonMaxSuppression, R2U_Net
from models.matching import OptimalMatching


def loadSample(name):
    window_size = 320
    image = io.imread(name)
    image = resize(
        image,
        (window_size, window_size, 3),
        anti_aliasing=True,
        preserve_range=True,
    )
    image = torch.from_numpy(image)
    image = image.permute(2, 0, 1) / 255.0
    return torch.unsqueeze(image, 0).float()


def split(points):
    points = np.array(points).flatten()
    even_locations = np.arange(points.shape[0] / 2) * 2
    odd_locations = even_locations + 1

    def pre(pts):
        return np.take(points, pts.tolist()).tolist()

    return (pre(even_locations), pre(odd_locations))


def bounding_box_from_points(X, Y):
    bbox = [min(X), min(Y), max(X) - min(X), max(Y) - min(Y)]
    bbox = [int(b) for b in bbox]
    return bbox


def single_annotation(poly):
    X, Y = split(poly)
    return {
        "segmentation": poly,
        "X": X,
        "Y": Y,
        "bbox": bounding_box_from_points(X, Y),
    }


def load():
    # Load network modules
    model = R2U_Net()
    model = model.train()

    head_ver = DetectionBranch()
    head_ver = head_ver.train()

    suppression = NonMaxSuppression()

    matching = OptimalMatching()
    matching = matching.train()

    # NOTE: The modules are set to .train() mode during inference to make sure that the BatchNorm layers
    # rely on batch statistics rather than the mean and variance estimated during training.
    # Experimentally, using batch stats makes the network perform better during inference.

    print("Loading pretrained model")
    model.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_backbone", map_location=torch.device("cpu")
        )
    )
    head_ver.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_seg_head", map_location=torch.device("cpu")
        )
    )
    matching.load_state_dict(
        torch.load(
            "./trained_weights/polyworld_matching", map_location=torch.device("cpu")
        )
    )

    # Initiate the dataloader
    def predict(filename):
        rgb = loadSample(filename)
        features = model(rgb)
        occupancy_grid = head_ver(features)

        _, graph_pressed = suppression(occupancy_grid)
        predictions = []
        poly = matching.predict(rgb, features, graph_pressed)

        for i, pp in enumerate(poly):
            for p in pp:
                predictions.append(single_annotation([p]))

            return predictions

    return predict


predict = load()
if __name__ == "__main__":
    print(predict("0.png"))
