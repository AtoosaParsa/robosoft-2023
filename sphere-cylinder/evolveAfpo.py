from afpo import AFPO
import constants as c
import random
import subprocess as sub

# Multiobjective AFPO
# have to set the number of fitnesses on line 34
# the fitnesses are maximized
# age is minimized

sub.call("rm -rf Data", shell=True)
sub.call("mkdir Data/", shell=True)

runs = c.RUNS
for r in range(1, runs+1):
    print("*********************************************************", flush=True)
    print("run: "+str(r), flush=True)
    randomSeed = r
    random.seed(r)
    afpo = AFPO(randomSeed, 2)
    afpo.evolve()
    