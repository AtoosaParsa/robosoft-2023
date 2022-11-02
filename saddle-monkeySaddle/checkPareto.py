import pickle
import matplotlib.pyplot as plt
from genome import GENOME
import constants as c
import numpy as np

runs = c.RUNS
gens = c.GENS

# plot the pareto front
# plot the avg vs generation plots

def paretoFront(population):
    pareto_front = []

    for i in population:
        i_is_dominated = False
        for j in population:
            if i != j:
                if population[j].dominates(population[i]):
                    i_is_dominated = True
                    break
        if not i_is_dominated:
            pareto_front.append(population[i])

    return pareto_front

paretoSize = []
for gen in range(2, gens):
    with open(f'Results/population/population{gen-1}.dat', "rb") as f:
        for r in range(1, runs+1):
            # population of the last generation
            temp = pickle.load(f)
            pf = paretoFront(temp)
            paretoSize.append((len(pf)))
            temp = []
    f.close()

plt.figure(figsize=(4,4))
plt.scatter(x=np.arange(1, gens-1, 1, dtype=int), y=paretoSize, color='blue', alpha=0.5)

plt.ylabel('pareto front size', fontsize=16)
plt.xlabel('generations', fontsize=16)
plt.title("Pareto Front", fontsize=16)
plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.tight_layout()
plt.show()
