import numpy as np
import scipy as sp
import random

# Measurements for the bounding box of the convex hulls

length = 1 # x-axis
depth = 1 # y-axis
height = 1 # z-axis
center = (0, 0, 0) # TEMP

for i in range



bounding_x = (-length/2, length/2)
bounding_y = (-depth/2, depth/2)
bounding_z = (height/2, depth/2)

x_coordinates = [0]
y_coordinates = []
z_coordinates = []
for _ in range(n_points):
    x = random.uniform(*bounding_x)
    y = random.uniform(*bounding_y)
    z = random.uniform(*bounding_z)

    x_coordinates.append(x)
    y_coordinates.append(y)
    z_coordinates.append(z)

