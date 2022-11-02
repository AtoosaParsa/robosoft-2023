import pickle
import matplotlib.pyplot as plt
import constants as c
import numpy

indices = numpy.arange(5, 609, 20)

gens = c.GENS

min_err1 = []
max_err1 = []
min_err2 = []
max_err2 = []
for g in range(1, gens):
    with open(f'Results/population/population{g}.dat', "rb") as f:
        min_t1 = 10000
        max_t1 = 0
        min_t2 = 10000
        max_t2 = 0
        population = pickle.load(f)
        ind = -1
        for p in population:
            if population[p].error1 < min_t1:
                min_t1 = population[p].error1
            if population[p].error2 < min_t2:
                min_t2 = population[p].error2
                ind = p
            if population[p].error1 > max_t1:
                max_t1 = population[p].error1
            if population[p].error2 > max_t2:
                max_t2 = population[p].error2
        if g == 299:
            if ind != -1:
                print(population[ind].ID)
                print(population[ind].error2/len(indices))
        min_err1.append(min_t1/len(indices))
        min_err2.append(min_t2/len(indices))
        max_err1.append(max_t1/len(indices))
        max_err2.append(max_t2/len(indices))

print(min_err1)
print(len(min_err1))
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
plt.plot(list(range(1, gens)), min_err1, color='blue', linestyle='dotted', alpha=0.6)
plt.plot(list(range(1, gens)), max_err1, color='blue', linestyle='dotted', alpha=0.6)

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