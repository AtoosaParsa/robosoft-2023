import constants as c
import copy
import numpy as np
import operator
import statistics
import pickle
import time
import random
from copy import deepcopy
import subprocess as sub

from genome import GENOME 

class AFPO:

    def __init__(self, randomSeed, num_fitnesses):        
        self.randomSeed = randomSeed
        self.currentGeneration = 0
        self.num_fitnesses = num_fitnesses
        self.nextAvailableID = 0


        self.genomes = {}

        for populationPosition in range(c.POP_SIZE):
            self.genomes[populationPosition] = GENOME(self.num_fitnesses, self.nextAvailableID)
            self.nextAvailableID = self.nextAvailableID + 1
        

    def evolve(self):
        self.performFirstGen(self.currentGeneration)
        for self.currentGeneration in range(1, c.GENS):
            self.performOneGen(self.currentGeneration)
            # making a checkpoint every 100 gens - save the whole population
            #if (self.currentGeneration % 100 == 0):
            self.SaveGen(self.currentGeneration)
        self.SaveLastGen(self.currentGeneration)

    def aging(self):
        for genome in self.genomes:
            self.genomes[genome].aging()


    def reducePopulation(self):
        pf = self.paretoFront()
        pf_size = len(pf)

        print("pf_size: "+str(pf_size), flush=True)

        counter = 0

        # remove individuals until the population size is popSize or size of the pareto front
        while len(self.genomes) > c.POP_SIZE: #max(pf_size, c.popSize):

            #print(counter, flush=True)
            counter = counter + 1
            
            pop_size = len(self.genomes)
            ind1 = np.random.randint(pop_size)
            ind2 = np.random.randint(pop_size)
            while ind1 == ind2:
                ind2 = np.random.randint(pop_size)

            # find the dominated genome and remove it
            if self.genomes[ind1].dominates(self.genomes[ind2]):  # ind1 dominates
                for i in range(ind2, len(self.genomes) - 1):
                    self.genomes[i] = self.genomes.pop(i + 1)

            elif self.genomes[ind2].dominates(self.genomes[ind1]):  # ind2 dominates
                for i in range(ind1, len(self.genomes) - 1):
                    self.genomes[i] = self.genomes.pop(i + 1)

            elif pf_size > c.POP_SIZE:
                # if the distance between the two genomes is small, keep the one with higher nandness
                if self.genomes[ind1].distance(self.genomes[ind2]):
                    print("there are close genomes!", flush=True)
                    if self.genomes[ind1].metric >= self.genomes[ind2].metric: # keep ind1
                        for i in range(ind2, len(self.genomes) - 1):
                            self.genomes[i] = self.genomes.pop(i + 1)
                    else: # keep ind2
                        for i in range(ind1, len(self.genomes) - 1):
                            self.genomes[i] = self.genomes.pop(i + 1)
                
                elif counter > 2*c.POP_SIZE: # to prevent the loop to go forever, if we've been here for a while
                    print("pareto front saturating, try increasing the popSize", flush=True) 
                    for i in range(ind2, len(self.genomes) - 1): # just randomly keep one of the two
                        self.genomes[i] = self.genomes.pop(i + 1)



    def evaluateAll(self, gen):
        print("evaluateing")
        for genome in self.genomes:
            self.genomes[genome].evaluate(gen)
        print("VXD files have been made")
        while True:
            print("simulating sheets")
            try:
                # simulate the robot
                sub.call("./voxcraft-sim -i Data/data{0}/ -o Data/data{1}/results.xml".format(gen, gen), shell=True) # -f > robot{2}.history
                break

            except IOError:
                print("Dang it! There was an IOError. I'll re-simulate this batch again...")
                pass

            except IndexError:
                print ("Shoot! There was an IndexError. I'll re-simulate this batch again...")
                pass
        print("simulating done, now compute fitness")   	
        for genome in self.genomes:
            self.genomes[genome].Compute_Fitness(gen)
            
    def evaluateInput(self, individuals, gen):
        print("evaluateing")
        for indv in individuals:
            indv.evaluate(gen)
        print("VXD files have been made")
        while True:
            print("simulating sheets")
            try:
                # simulate the robot
                sub.call("./voxcraft-sim -i Data/data{0}/ -o Data/data{1}/results.xml".format(gen, gen), shell=True) # -f > robot{2}.history
                break

            except IOError:
                print("Dang it! There was an IOError. I'll re-simulate this batch again...")
                pass

            except IndexError:
                print ("Shoot! There was an IndexError. I'll re-simulate this batch again...")
                pass
        print("simulating done, now compute fitness")   	
        for indv in individuals:
            indv.Compute_Fitness(gen)
        
        return individuals

    def increasePopulation(self, children):
        popSize = len(self.genomes)
        j = 0
        for i in range(popSize , 2*popSize):
            self.genomes[i] = children[j]
            j = j + 1

    def breeding(self):
        # increase the population to 2*popSize-1 by adding random genomes and mutating them
        popSize = len(self.genomes)
        children = []
        for newGenome in range(popSize , 2*popSize-1):
            randGen = np.random.randint(popSize)
            child = copy.deepcopy(self.genomes[randGen])
            child.mutate()
            child.ID = self.nextAvailableID
            self.nextAvailableID = self.nextAvailableID + 1
            children.append(child)
        return children

    def findBestGenome(self):
        best = None
        for g in self.genomes:
            if best is None:
                best = self.genomes[g]
            if self.genomes[g] is not None and self.genomes[g].dominatesAll(best):
                best = self.genomes[g]
        return best

    #def findAvgFitness(self):
    #    add = np.zeros(self.num_fitnesses)   
    #    for g in self.genomes:
    #        add += np.array(self.genomes[g].fitnesses)
    #    return  np.round(add/len(self.genomes), decimals=2)
    
    def injectOne(self, children):
        # add one ranom genome
        children.append(GENOME(self.num_fitnesses, self.nextAvailableID))
        self.nextAvailableID = self.nextAvailableID + 1
        return children

    def performFirstGen(self, gen):
        self.evaluateAll(gen)
        self.printing()
        self.saveBest(gen)
        self.saveAvg()

    def performOneGen(self, gen):
        print("pop size: "+str(len(self.genomes)), flush=True)
        print("aging", flush=True)
        self.aging()
        print("breeding", flush=True)
        children = self.breeding()
        print("injecting", flush=True)
        children = self.injectOne(children)
        print("evaluateing", flush=True)
        children = self.evaluateInput(children, gen)
        print("entend the population")
        self.increasePopulation(children)
        print("reducing", flush=True)
        self.reducePopulation()
        print("Printing", flush=True)
        self.printing()
        print("Saving best", flush=True)
        self.saveBest(gen)
        print("save avg", flush=True)
        self.saveAvg()

    def printing(self):
        print('Generation ', end='', flush=True)
        print(self.currentGeneration, end='', flush=True)
        print(' of ', end='', flush=True)
        print(str(c.GENS), end='', flush=True)
        print(': ', end='', flush=True)

        bestGenome = self.findBestGenome()
        bestGenome.genomePrint()

    def SavePopulation(self, gen):
        f = open(f'Data/population{gen}.dat', 'ab')
        pickle.dump(self.genomes , f)
        f.close()

    def saveBest(self, gen):
        bestGenome = self.findBestGenome()
        bestGenome.SaveBest(gen)
    
    def SaveLastGen(self, gen):
        f = open(f'Data/population{gen}.dat', 'ab')
        pickle.dump(self.genomes, f)
        f.close()

    def SaveGen(self, gen):
        f = open(f'Data/population{gen}.dat', 'ab')
        pickle.dump(self.genomes, f)
        f.close()

    def saveAvg(self):
        f_sum = 0
        for g in self.genomes:
            f_sum += self.genomes[g].error1 
        avg = f_sum / len(self.genomes)
        
        f = open(f'Data/avg_error1.dat', 'ab')
        pickle.dump(avg , f)
        f.close()

        f_sum = 0
        for g in self.genomes:
            f_sum += self.genomes[g].error2
        avg = f_sum / len(self.genomes)
        
        f = open(f'Data/avg_error2.dat', 'ab')
        pickle.dump(avg , f)
        f.close()

        return avg
        
    #def showBestGenome(self):
    #    bestGenome = self.findBestGenome()
    #    bestGenome.genomeShow()

    def paretoFront(self):
        pareto_front = []

        for i in self.genomes:
            i_is_dominated = False
            for j in self.genomes:
                if i != j:
                    if self.genomes[j].dominates(self.genomes[i]):
                        i_is_dominated = True
            if not i_is_dominated:
                pareto_front.append(self.genomes[i])

        return pareto_front
