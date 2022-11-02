import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import subprocess as sub

DIM = 28*2
DIAMETER = 0.14
# size of one voxel
vxSize = DIAMETER / DIM

center = DIM / 2.0 - 1
radius = DIM / 2.0

center = 0
radius = radius * vxSize * 1000

def normalize(x, minx, maxx, a, b):
    return (b - a)*((x - minx)/(maxx - minx)) + a

# sphere
ax = plt.figure().add_subplot(projection='3d')

X = np.arange(-70, 70, 2.5)
Y = np.arange(-70, 70, 2.5)
X, Y = np.meshgrid(X, Y)
Z = -1* np.sqrt(np.power(radius, 2) - np.power(X, 2) - np.power(Y, 2))

for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) >= radius):
            Z[i, j] = np.nan

a = -15#0
b = 15#18
minz = np.min(Z)
maxz = np.max(Z)

Z = np.array(Z, dtype=float)

surf = ax.plot_surface(X, Y, Z, alpha=0.5, color='red')#cmap='hot')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(color='white')
plt.yticks(color='white')
plt.xlim([-70, 70])
#plt.title("Sphere", fontsize=14)
#plt.tight_layout()
#ax.set_xlabel("x (mm)", fontsize=12)
#ax.set_ylabel("y (mm)", fontsize=12)
#ax.set_zlabel("z (mm)", fontsize=12)
#plt.axis('off')
plt.show()

# simple saddle
ax = plt.figure().add_subplot(projection='3d')

X = np.arange(-70, 70, 2.5)
Y = np.arange(-70, 70, 2.5)
X, Y = np.meshgrid(X, Y)
Z = -1 * np.sqrt(np.power(radius, 2) - np.power(X, 2))

for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) >= radius):
            Z[i, j] = np.nan
            
a = -15#0
b = 15#17
minz = np.min(Z)
maxz = np.max(Z)



Z = np.array(Z, dtype=float)

surf = ax.plot_surface(X, Y, Z, alpha=0.5, color='red')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(color='white')
plt.yticks(color='white')
plt.show()