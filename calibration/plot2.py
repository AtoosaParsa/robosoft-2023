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

# in mms
realZs_2 = np.array([-2, -1, 2, -1, 2, 2, 2, 2, 2, 3, 4, 1, 1, 3, 1, 1])
realZs_2_5 = np.array([-2, -1, 5, 1, 2, 5, 4, 1, 0, 6, 5, 1, 1, 7, 1, 1])
realZs_3 = np.array([-2, 1, 11, 0, 1, 12, 10, 1, 1, 12, 13, 2, 2, 14, 3, 1])
realZs_3_5 = np.array([-2, 0, 18, 1, 5, 19, 18, 5, 3, 18, 17, 3, 4, 19, -1, -2])

# this is unprocesses, have to subtract 585:
#realZs = np.subtract(realZs, 585)

# input the position in meters and get the index of the 50 by 50 grid
def getIndex(x, y, vxSize):
    # size of one voxel
    return int(x/vxSize), int(y/vxSize)

DIM = 28*2
DIAMETER = 0.14
center = DIM / 2.0 - 1
radius = DIM / 2.0

# size of one voxel
vxSize = DIAMETER / DIM

# get the data from simulation. This is also in mm rounded so the precision matched reality
f = open('target/simple.dat', 'rb')
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
surf = ax.plot_surface(X, Y, grid_heights)

#plt.show()

#fig = plt.figure(figsize=(4,4))
#ax = fig.add_subplot(projection='3d')

#ax.set_xlabel('$f_1(\mathbf{x})$'+' = O1', fontsize=14)
#ax.set_ylabel('$f_2(\mathbf{x})$'+' = O2', fontsize=14)
#ax.set_zlabel('$f_3(\mathbf{x})$'+' = O3', fontsize=14)
#ax.set_title("Pareto Front", fontsize=16)
# Get rid of the panes                          
ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0)) 
ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
#ax.xaxis.grid(color='skyblue', linestyle=':', linewidth=0.5)
#ax.yaxis.grid(color='skyblue', linestyle=':', linewidth=0.5)
#plt.xticks(fontsize=12)
#plt.yticks(fontsize=12)
#plt.zticks(fontsize=12)
plt.tight_layout()
plt.show()
