from ctypes.wintypes import SIZE
import numpy as np
import os
import math
import random
import subprocess as sub
from lxml import etree
import pickle

with open('targets/target_sphere.dat', "rb") as f:
    target1 = pickle.load(f)

with open('targets/target_cylinder.dat', "rb") as f:
    target2 = pickle.load(f)

with open("Results/data_1/results.xml", 'r') as f:
    tree = etree.parse(f)

robots = tree.xpath("//detail/*")

for robot in robots:
    print(robot.tag)
    temp1 = robot.xpath("init_pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions_i = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    positions_i_z = np.zeros(1941)
    for i in range(1941):
        positions_i_z[i] = positions_i[i, 2] + positions_i[i+1941, 2]
        positions_i_z[i] = positions_i_z[i] + positions_i[i+2*1941, 2] / 3
    
    temp1 = robot.xpath("pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions1 = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    positions1_z = np.zeros(1941)
    for i in range(1941):
        positions1_z[i] = positions1[i, 2] + positions1[i+1941, 2]
        positions1_z[i] = positions1_z[i] + positions1[i+2*1941, 2] / 3


n = 0
errs = 0
max_errs = 0
for i in range(len(target1)):
    error = abs(positions1_z[i] - target1[i])
    errs = errs + error
avg_err1 = errs

print("error1: " + str(avg_err1))

with open("Results/data_2/results.xml", 'r') as f:
    tree = etree.parse(f)

robots = tree.xpath("//detail/*")

for robot in robots:
    print(robot.tag)
    temp1 = robot.xpath("init_pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions_i = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    positions_i_z = np.zeros(1941)
    for i in range(1941):
        positions_i_z[i] = positions_i[i, 2] + positions_i[i+1941, 2]
        positions_i_z[i] = positions_i_z[i] + positions_i[i+2*1941, 2] / 3
    
    temp1 = robot.xpath("pos")[0].text.replace(';', ',')
    temp2 = np.fromstring(temp1, dtype=float, sep=',')
    positions1 = np.reshape(temp2, (int(np.size(temp2)/3), 3))
    positions1_z = np.zeros(1941)
    for i in range(1941):
        positions1_z[i] = positions1[i, 2] + positions1[i+1941, 2]
        positions1_z[i] = positions1_z[i] + positions1[i+2*1941, 2] / 3

n = 0
errs = 0
max_errs = 0
for i in range(len(target1)):
    error = abs(positions1_z[i] - target2[i])
    errs = errs + error
avg_err1 = errs

print("error2: " + str(avg_err1))

