#!/bin/env python

# Validate and plot some of the torus projections
# Dan Stowell 2025

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from tools import utils

np.set_printoptions(suppress=True)


#############################################
def array2string_local(X):
    return np.array2string(np.squeeze(X), precision=3, floatmode='fixed')

#############################################
# Generate some simple data for illustration & validation

qpointslin = np.linspace(0.0, 1.0, num=16)

# project the linspace onto a circle ("clifford embedding")
qpointslin_expanded = np.expand_dims(qpointslin, axis=1)
print(f"Linear points: {array2string_local(qpointslin_expanded)}")

qpointscirc = utils.circular_torus_embed(qpointslin_expanded)
print(f"Circ points: {array2string_local(qpointscirc)}")

# NOTE: this "unwrap" is only valid here BECAUSE it's a 1D torus.
# If it was more than 1D then we would need to reorder the dimensions first.
qpointscirc_unwrapped = utils.unwrap_pairwise_torus(qpointscirc, pandas=False)
print(f"Circu points: {array2string_local(qpointscirc_unwrapped)}")

deltas = qpointscirc_unwrapped - qpointslin_expanded
print(f"Circu points delta: {array2string_local(deltas)}")


#############################################
# Plot rainbow using meshgrid to illustrate the two types of embedding

qpointslin = np.arange(-1.125, 1.12501, 0.125 / 4)

# meshgrid
xv, yv = np.meshgrid(qpointslin, qpointslin)

coordsgrid = np.array([xv.flatten(), yv.flatten()]).T

cgu = utils.unwrap_pairwise_torus(coordsgrid, pandas=False).reshape(xv.shape)
# unwrapped, should be circular rainbow...


fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

ax1.imshow(cgu, cmap='hsv', extent=(qpointslin[0], qpointslin[-1], qpointslin[0], qpointslin[-1]),
    aspect='equal')


# We can reuse the meshgrids to make the modulo-wrapped values
xvt = xv % 1.0
ax2.imshow(xvt, cmap='hsv', extent=(qpointslin[0], qpointslin[-1], qpointslin[0], qpointslin[-1]),
    aspect='equal')
yvt = yv % 1.0
ax3.imshow(yvt, cmap='hsv', extent=(qpointslin[0], qpointslin[-1], qpointslin[0], qpointslin[-1]),
    aspect='equal')
plt.savefig("plots/plot_rainbows.pdf")


