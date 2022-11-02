import pickle
import matplotlib.pyplot as plt
from genome import GENOME
import constants as c
import numpy

runs = c.RUNS
gens = c.GENS

indices = numpy.arange(5, 609, 20)
# make output files for fabrication
final = 28979
num=2
with open(f'Results{num}/population{gens-1}.dat', "rb") as f:
    temp = pickle.load(f)
    for i in temp:
        if temp[i].ID == final:
            temp[i].SaveForExp(num)
            print("errors: " + str(temp[i].ID) + " :: " + str(-1*temp[i].fitnesses[0]/len(indices)) + ", "+str(-1*temp[i].fitnesses[1]/len(indices)))

f.close()
