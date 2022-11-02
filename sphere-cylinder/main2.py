import numpy as np
import os
import math
import random
import subprocess as sub
from lxml import etree

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD


sub.call("./voxcraft-sim -i Results/data_1/ -o Results/data_1/results.xml -f > Results/data_1/robot.history", shell=True)

sub.call("./voxcraft-sim -i Results/data_2/ -o Results/data_2/results.xml -f > Results/data_2/robot.history", shell=True)







