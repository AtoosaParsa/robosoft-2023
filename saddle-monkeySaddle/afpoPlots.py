import pickle
import matplotlib.pyplot as plt
from genome import GENOME
import constants as c
import numpy

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

with open(f'Results3/population{gens-1}.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        pf = paretoFront(temp)
        print("pf size: "+str(len(pf)))

        plt.figure(figsize=(4,4))
        counter = 0
        for i in pf:
            plt.scatter(x=-1*i.fitnesses[0], y=-1*i.fitnesses[1], color='blue', alpha=0.5)
            plt.annotate(str(i.ID), (-1*i.fitnesses[0], -1*i.fitnesses[1]))
            print(str(counter) + ": " + str(i.ID) + " :: " + str(-1*i.fitnesses[0]) + ", "+str(-1*i.fitnesses[1]))
            counter = counter + 1
        plt.ylabel('$f_2(\mathbf{x})$'+' = error2 (target = simple saddle)', fontsize=16)
        plt.xlabel('$f_1(\mathbf{x})$'+' = error1 (target = monkey saddle)', fontsize=16)
        plt.title("Pareto Front", fontsize=16)
        plt.grid(color='skyblue', linestyle=':', linewidth=0.5)
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.tight_layout()
        plt.show()

        temp = []
f.close()
