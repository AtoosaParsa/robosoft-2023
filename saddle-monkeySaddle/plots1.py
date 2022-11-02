import pickle
import matplotlib.pyplot as plt
import constants as c
import numpy

indices = numpy.arange(0, 609, 1)

gens = c.GENS
runs = 3

fitnesses = numpy.zeros([runs, gens])
temp = []
with open('Results1/avg_error1.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[2] = newTemp
    temp = []
f.close()
temp = []
with open('Results2/avg_error1.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[0] = newTemp
    temp = []
f.close()
temp = []
with open('Results3/avg_error1.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[1] = newTemp
    temp = []
f.close()

mean_f = numpy.mean(fitnesses, axis=0)
std_f = numpy.std(fitnesses, axis=0)

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), mean_f, color='blue', label="Monkey Saddle", linewidth=2)
plt.fill_between(list(range(1, gens+1)), mean_f-std_f, mean_f+std_f, color='cornflowerblue', alpha=0.3, linewidth=1)

fitnesses = numpy.zeros([runs, gens])
temp = []
with open('Results1/avg_error2.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[2] = newTemp
    temp = []
f.close()

with open('Results2/avg_error2.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[0] = newTemp
    temp = []
f.close()

with open('Results3/avg_error2.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    newTemp = [x / len(indices) for x in temp]
    fitnesses[1] = newTemp
    temp = []
f.close()

mean_f = numpy.mean(fitnesses, axis=0)
std_f = numpy.std(fitnesses, axis=0)
plt.plot(list(range(1, gens+1)), mean_f, color='red', label="Saddle", linewidth=2)
plt.fill_between(list(range(1, gens+1)), mean_f-std_f, mean_f+std_f, color='lightcoral', alpha=0.3, linewidth=1)

plt.xlabel("Generations", fontsize=32)
plt.ylabel("Average Error (mm)", fontsize=32)
plt.title("Evolutionary Search", fontsize=32)
plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
plt.minorticks_on()
plt.xticks(fontsize=28)
plt.yticks(fontsize=28)
plt.rc('legend', fontsize=32)
plt.tight_layout()
plt.legend(loc='upper right', )
plt.show()
