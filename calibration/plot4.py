import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math

import pickle

# in cms
realXs = np.array([12.5, 12.5, 10.5, 10.5, 10.5, 10.5, 7, 7, 7, 7, 3.5, 3.5, 3.5, 3.5, 1.75, 1.75])
realYs = np.array([8.75, 5.25, 12.25, 8.75, 5.25, 1.75, 12.25, 8.75, 5.25, 1.75, 12.25, 8.75, 5.25, 1.75, 8.75, 5.25])

# convert to meters
realXs = np.divide(realXs, 100)
realYs = np.divide(realYs, 100)

voxelNum = 609#2449
DIM = 28#*2
DIAMETER = 0.14
center = DIM / 2.0 - 1
radius = DIM / 2.0
# size of one voxel
vxSize = DIAMETER / DIM

# get the data from simulation. This is also in mm rounded so the precision matched reality
f = open('target/simpleSaddle.dat', 'rb')
heights = pickle.load(f)
f.close()

grid_heights = np.zeros((DIM, DIM))
index = 0
for x in range(DIM):
  for y in range(DIM):
    if (math.sqrt(pow(x-center, 2.0)+pow(y-center, 2.0)) < radius):
      grid_heights[x, y] = heights[index]
      index = index + 1
    else:
      grid_heights[x, y] = np.nan

for i in range(DIM):
  print(grid_heights[i, :])

ax = plt.figure().add_subplot(projection='3d')

X = np.arange(0, DIAMETER*1000, DIAMETER*1000/DIM)
Y = np.arange(0, DIAMETER*1000, DIAMETER*1000/DIM)
X, Y = np.meshgrid(X, Y)
surf = ax.plot_surface(X, Y, grid_heights, alpha=0.5, label='target', color='red')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d

#for i in range(len(realXs)):
#    print(str(realZs_2[i]) + " :: " + str(grid_heights[int(realYs[i]/vxSize), int(realXs[i]/vxSize)]))
#    error = error + abs(realZs_2[i] - grid_heights[int(realYs[i]/vxSize), int(realXs[i]/vxSize)])
#err1 = error
#print(error)

ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
ax.legend(loc='upper right')
ax.set_xlabel("x (mm)", fontsize=18)
ax.set_ylabel("y (mm)", fontsize=18)
ax.set_zlabel("z (mm)", fontsize=18)
plt.tight_layout()
plt.show()
