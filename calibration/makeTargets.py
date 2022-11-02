import matplotlib.pyplot as plt
import numpy as np
import math
import pickle
import subprocess as sub

sub.call("rm -rf target", shell=True)
sub.call("mkdir target/", shell=True)

def normalize(x, minx, maxx, a, b):
    return (b - a)*((x - minx)/(maxx - minx)) + a

# monkey saddle
ax = plt.figure().add_subplot(projection='3d')

DIM = 28
DIAMETER = 0.14
# size of one voxel
vxSize = DIAMETER / DIM

center = DIM / 2.0 - 1
radius = DIM / 2.0

center = 0
radius = radius * vxSize * 1000

X = np.arange(-70, 70, 5)
Y = np.arange(-70, 70, 5)
X, Y = np.meshgrid(X, Y)
Z = np.power(X, 3) - 3*X*np.power(Y, 2)
Z = np.array(Z, dtype=float)

for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) >= radius):
            Z[i, j] = np.nan

a = -20#0
b = 20#18
minz = np.nanmin(Z)
maxz = np.nanmax(Z)

z_monkey = []

for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) < radius):
            Z[i, j] = normalize(Z[i, j], minz, maxz, a, b)
            z_monkey.append(Z[i, j].round(0))
        else:
            Z[i, j] =np.nan


f = open('target/monkeySaddle.dat', 'ab')
pickle.dump(z_monkey, f)
f.close()

surf = ax.plot_surface(X, Y, Z, edgecolors='grey', lw=0.5, rstride=1, cstride=1, alpha=0.9, cmap='hot')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Monkey Saddle", fontsize=14)
plt.tight_layout()
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_zlabel("z (mm)", fontsize=12)
plt.show()

# simple saddle
ax = plt.figure().add_subplot(projection='3d')

X = np.arange(-70, 70, 5)
Y = np.arange(-70, 70, 5)
X, Y = np.meshgrid(X, Y)
Z = np.power(X, 2) - np.power(Y, 2)

Z = np.array(Z, dtype=float)
for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) >= radius):
            Z[i, j] = np.nan

minz = np.nanmin(Z)
maxz = np.nanmax(Z)

a = -20#0
b = 20#17


Z = np.array(Z, dtype=float)
z_saddle = []

for i in range(len(X)):
    for j in range(len(Y)):
        if (math.sqrt(pow(X[i, j]-center, 2.0)+pow(Y[i, j]-center, 2.0)) < radius):
            Z[i, j] = normalize(Z[i, j], minz, maxz, a, b)
            z_saddle.append(Z[i, j].round(0))
        else:
            Z[i, j] = np.nan

f = open('target/simpleSaddle.dat', 'ab')
pickle.dump(z_saddle, f)
f.close()

surf = ax.plot_surface(X, Y, Z, edgecolors='grey', lw=0.5, rstride=1, cstride=1, alpha=0.9, cmap='hot')
surf._facecolors2d=surf._facecolor3d
surf._edgecolors2d=surf._edgecolor3d
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
plt.grid(which='major', color='lightblue', linestyle=':', linewidth=0.05, alpha=0.1)
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.title("Simple Saddle", fontsize=14)
plt.tight_layout()
ax.set_xlabel("x (mm)", fontsize=12)
ax.set_ylabel("y (mm)", fontsize=12)
ax.set_zlabel("z (mm)", fontsize=12)
plt.show()