import pickle
import matplotlib.pyplot as plt
import constants as c
import numpy

indices = numpy.arange(5, 609, 20)

gens = c.GENS
fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results/avg_error1.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()

newFitnesses = [x / len(indices) for x in fitnesses]
plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), newFitnesses, color='blue', label="error1 (target=cylinder)")

fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results/avg_error2.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()
newFitnesses = [x / len(indices) for x in fitnesses]
plt.plot(list(range(1, gens+1)), newFitnesses, color='red', label="error2 (target=sphere)")

plt.xlabel("Generations", fontsize=18)
plt.ylabel("Error", fontsize=18)
plt.title("Evolutionary Search", fontsize=18)
plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
plt.tight_layout()
plt.legend(loc='upper right', )
plt.show()

fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results/avg_fitness.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(-1*pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()

plt.figure(figsize=(6.4,4.8))
plt.plot(list(range(1, gens+1)), fitnesses, color='blue', label="average fitness")

fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results/best_fitness.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(-1*pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()

plt.plot(list(range(1, gens+1)), fitnesses, color='red', label="best fitness")

plt.xlabel("Generations", fontsize=18)
plt.ylabel("Fitness", fontsize=18)
plt.title("Evolutionary Search", fontsize=18)
plt.grid(color='skyblue', linestyle=':', linewidth=0.75)
plt.tight_layout()
plt.legend(loc='upper right', )
plt.show()
