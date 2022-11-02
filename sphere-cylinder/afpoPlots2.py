import pickle
import matplotlib.pyplot as plt
from genome import GENOME
import constants as c
import numpy
import operator

runs = c.RUNS
gens = c.GENS

# plot the pareto front
# plot the avg vs generation plots

def normalize(x, minx, maxx, a, b):
    return (b - a)*((x - minx)/(maxx - minx)) + a

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

indices = numpy.arange(5, 609, 20)

fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results1/avg_error1.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()

avgError1 = fitnesses[0]/len(indices)
print(avgError1)

fitnesses = numpy.zeros(gens)
temp = []
individuals = []
with open('Results1/avg_error2.dat', "rb") as f:
    for g in range(1, gens+1):
        try:
            temp.append(pickle.load(f))
        except EOFError:
            break
    fitnesses = temp
    temp = []
f.close()

avgError2 = fitnesses[0]/len(indices)

with open(f'Results1/population{gens-1}.dat', "rb") as f:
    for r in range(1, runs+1):
        # population of the last generation
        temp = pickle.load(f)
        pf = paretoFront(temp)
        print("pf size: "+str(len(pf)))

        plt.figure(figsize=(4,4))
        err1 = []
        err2 = []
        colors = []
        ages = []
        for i in pf:
            err1.append(-1*i.fitnesses[0]/len(indices))
            err2.append(-1*i.fitnesses[1]/len(indices))
            ages.append(i.age)
        colors = (err1-avgError1) + (err2-avgError2)
        sortedList = list(zip(*sorted(zip(pf,colors), key=operator.itemgetter(1)))) #sortedList[0:genome, 1:colors][0...len]
        print(sortedList[0][0].ID)
        #count = 0
        #for i in pf:
        #    print(str(colors[count]) + ": "+ str(i.ID))
        #    count = count + 1
        #print(colors)
        #minn = min(colors)
        #maxx = max(colors)
        #for i in range(len(colors)):
        #    colors[i] = normalize(colors[i], minn, maxx, 1, 10)
        #print(colors)
        plt.scatter(x=err1, y=err2, c=colors, s=100, edgecolors='black', linewidth = 0.5, cmap='PiYG') #gnuplot
        #for i in range(len(ages)):
        #    plt.annotate(" "+str(ages[i]), (err1[i], err2[i]))
        plt.vlines(x=avgError1, ymin=0, ymax=max(max(err1), max(err2))+0.5, linewidth=2, linestyle='--', color='red', alpha=0.9)
        #plt.annotate(str(avgError1.round(2)), (avgError1+0.1, max(err2)), fontsize=12, color='red')
        plt.hlines(y=avgError2, xmin=0, xmax=max(max(err1), max(err2))+0.5, linewidth=2, linestyle='--', color='red', alpha=0.9)
        #plt.annotate(str(avgError2.round(2)), (max(err1), avgError2+0.1), fontsize=12, color='red')
        plt.ylabel('Sphere Error', fontsize=28) #'$f_2(\mathbf{x})$'+' = error2 (target=sphere)'
        plt.xlabel('Cylinder Error', fontsize=28) #'$f_1(\mathbf{x})$'+' = error1 (target=cylinder)'
        plt.title("Pareto Front", fontsize=28)
        plt.grid(which='minor', color='skyblue', linestyle=':', linewidth=0.3)
        plt.grid(which='major', color='skyblue', linestyle='-', linewidth=0.5)
        plt.minorticks_on()
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.tight_layout()
        plt.xlim([0, max(max(err1), max(err2))+0.5])
        plt.ylim([0, max(max(err1), max(err2))+0.5])
        cbar=plt.colorbar()
        cbar.set_label("L1 norm", fontsize=24)
        cbar.ax.tick_params(labelsize=24)
        # the chosen solution from the pareto front
        #plt.scatter(x=sortedList[0][0].error1/len(indices), y=sortedList[0][0].error2/len(indices), facecolors='none', edgecolors='magenta', alpha=1, linewidth=0.8, s=150, marker='s')
        plt.show()

        print("error1"+str(avgError1.round(2)))
        print("error2"+str(avgError2.round(2)))
        temp = []
f.close()
