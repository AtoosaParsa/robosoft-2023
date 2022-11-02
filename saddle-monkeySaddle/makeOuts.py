import pickle
import matplotlib.pyplot as plt
from genome import GENOME
import constants as c
import numpy

runs = c.RUNS
gens = c.GENS

indices = numpy.arange(0, 609, 1)
# make output files for fabrication
final = 24178
num = 3
with open(f'Results{num}/population{gens-1}.dat', "rb") as f:
    temp = pickle.load(f)
    for i in temp:
        if temp[i].ID == final:
            temp[i].SaveForExp(num)
            print("errors: " + str(temp[i].ID) + " :: " + str(-1*temp[i].fitnesses[0]/len(indices)) + ", "+str(-1*temp[i].fitnesses[1]/len(indices)))

f.close()
