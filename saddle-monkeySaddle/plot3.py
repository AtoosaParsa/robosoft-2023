import matplotlib.pyplot as plt
import matplotlib.colors
import numpy as np
import math

import pickle

from lxml import etree
filename = "Results1/data_2/results.xml"
with open(filename, 'r') as f:
    tree = etree.parse(f)

voxelNum = 609 #2449 #609#

robots = tree.xpath("//detail/*")
for robot in robots:
    print(robot.tag)
    temp1 = robot.xpath("init_pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions_i = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    print(np.shape(positions_i))
    positions_i_z = np.zeros(voxelNum)
    for i in range(voxelNum):
        positions_i_z[i] = positions_i[i, 2] + positions_i[i+voxelNum, 2]
        positions_i_z[i] = (positions_i_z[i] + positions_i[i+2*voxelNum, 2]) / 3

    temp1 = robot.xpath("pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions1 = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    positions1_z = np.zeros(voxelNum)
    for i in range(voxelNum):
        positions1_z[i] = positions1[i, 2] + positions1[i+voxelNum, 2]
        positions1_z[i] = (positions1_z[i] + positions1[i+2*voxelNum, 2]) / 3
#print(1000.*positions1[:, 2])
#heights = ((positions1_z - positions_i_z)*1000).round(0)
# just use the heights of the first layer - for the uninflated, this is zero
heights = np.multiply(1000, positions1[0:voxelNum, 2]).round(0)
print(heights)

DIM = 28#*2
DIAMETER = 0.14
center = DIM / 2.0 - 1
radius = DIM / 2.0

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
surf = ax.plot_surface(X, Y, grid_heights, alpha=1, rstride=1, cstride=1, color='red', label='optimized', cmap='Reds')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#ax.legend(loc='upper right')
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_zlabel("z (mm)", fontsize=12)
plt.show()

ax = plt.figure().add_subplot(projection='3d')
# get the data from simulation. This is also in mm rounded so the precision matched reality
f = open('targets/simpleSaddle.dat', 'rb')
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

surf = ax.plot_surface(X, Y, grid_heights, alpha=0.9, rstride=1, cstride=1, color ='blue', label='target', cmap='Blues')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.tight_layout()
#ax.legend(loc='upper right')
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_zlabel("z (mm)", fontsize=12)
plt.show()
