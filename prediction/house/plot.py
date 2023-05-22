#!/usr/bin/env python3
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import prediction as p

fn = sys.argv[1]
result = p.predict(fn)
img = np.asarray(Image.open(fn).resize((300, 300)))

plt.imshow(img)


def fix(xs):
    s = 3.0/3.0
    return [x * s for x in xs]


for e in result:
    plt.plot(fix(e["X"]), fix(e["Y"]), marker="o")
axes = plt.gca()
axes.set_xlim([0, 300])
axes.set_ylim([0, 300])

plt.show()
