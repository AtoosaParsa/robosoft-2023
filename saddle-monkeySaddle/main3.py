import numpy as np
import os
import math
import random
import subprocess as sub
from lxml import etree

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD


sub.call("./voxcraft-sim -i Results/data_3/ -o Results/data_3/results.xml -f > Results/data_3/robot.history", shell=True)

sub.call("./voxcraft-sim -i Results/data_4/ -o Results/data_4/results.xml -f > Results/data_4/robot.history", shell=True)

sub.call("./voxcraft-sim -i Results/data_5/ -o Results/data_5/results.xml -f > Results/data_5/robot.history", shell=True)

sub.call("./voxcraft-sim -i Results/data_6/ -o Results/data_6/results.xml -f > Results/data_6/robot.history", shell=True)





