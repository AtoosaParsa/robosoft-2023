from Bezier import Bezier
import constants as c
import random
import math
import numpy as np
import sys
import subprocess as sub
import os
from lxml import etree
import pickle
import copy

from VoxcraftVXA import VXA
from VoxcraftVXD import VXD

class GENOME:

    def __init__(self, num_fitnesses, id):
        self.genome1 = np.zeros((c.FIBERS*2, 4)) # 6 fibers on top, 6 on the bottom, by starting and ending points (sx, sy, ex, ey)
        self.genome2 = np.zeros(4) #top-c1, bot-c1, top-c2, bot-c2

        self.indices = np.arange(5, 609, 20)
        # initialize genome2
        self.genome2[0] = np.random.randint(0, high=c.FIBERS+1, dtype='int')
        self.genome2[2] = np.random.randint(0, high=c.FIBERS+1, dtype='int')
        while (self.genome2[0]+self.genome2[2] > c.FIBERS): # maximum 6 fibers in total on the top
            self.genome2[2] = np.random.randint(0, high=c.FIBERS+1, dtype='int')
        self.genome2[1] = np.random.randint(0, high=c.FIBERS+1, dtype='int')
        self.genome2[3] = np.random.randint(0, high=c.FIBERS+1, dtype='int')
        while (self.genome2[1]+self.genome2[3] > c.FIBERS): # maximum 6 fibers in total on the bottom
            self.genome2[3] = np.random.randint(0, high=c.FIBERS+1, dtype='int')

        # initialize genome1
        temp = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
        t_points = np.arange(0, 1, 0.01) # for descritization
        for i in range(2*c.FIBERS): # now let's make random starting and ending points
            if i > c.FIBERS-1:
                temp = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # reset the grid keeping track of intersections becasue we just moved to the bottom sheet
            allCheck = False
            while not allCheck:
                points = np.random.randint(0, high=c.GRID_SIZE, size=4, dtype='int')
                # constraint#1: make sure the points have the minimum length
                check1 = self.checkLength(points)
                # constraint#2: make sure the points are inside the circular sheet
                check2 = self.checkCircle(points)
                # constraint#3: check the number of intersections
                points_ = np.array([[points[0], points[1]], [points[2], points[3]]])
                curve = Bezier.Curve(t_points, points_)
                grid = Bezier.DistCurve(curve)
                gridCheck = temp + grid
                check3 = np.all(gridCheck<3)
                allCheck = np.all([check1, check2, check3])
            # save the point to the genome
            self.genome1[i, :] = points
            # moving on to the next point, update the grid with the accepted fiber
            temp = temp + grid

        self.error1 = 10000
        self.error2 = 10000
        self.overlap = 10000
        self.ID = id
        self.top_grid_c1 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        self.bot_grid_c1 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        self.top_grid_c2 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        self.bot_grid_c2 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))

        self.age = 0
        self.fitnesses = [-1000.0 for x in range(num_fitnesses)]
        self.metric = 0
        # determines if this individual was already evaluated
        self.needs_eval = True

        # number of voxels in one layer
        self.voxelNum = 609

    def aging(self):
        self.age = self.age + 1

    def dominates(self, other):
        # returns True if self dominates other param other, False otherwise.

        self_min_traits = self.get_minimize_vals()
        self_max_traits = self.get_maximize_vals()

        other_min_traits = other.get_minimize_vals()
        other_max_traits = other.get_maximize_vals()

        # all min traits must be at least as small as corresponding min traits
        if list(filter(lambda x: x[0] > x[1], zip(self_min_traits, other_min_traits))):
            return False

        # all max traits must be at least as large as corresponding max traits
        if list(filter(lambda x: x[0] < x[1], zip(self_max_traits, other_max_traits))):
            return False

        # any min trait smaller than other min trait
        if list(filter(lambda x: x[0] < x[1], zip(self_min_traits, other_min_traits))):
            return True

        # any max trait larger than other max trait
        if list(filter(lambda x: x[0] > x[1], zip(self_max_traits, other_max_traits))):
            return True

        # all fitness values are the same, default to return False.
        return self.ID < other.ID

    def distance(self, other):
        # checks the distance in genotype space, if they are close enough, returns true
        dist = np.array([abs(self.fitnesses[0] - other.fitnesses[0]), abs(self.fitnesses[1] - other.fitnesses[1])])
        threshold = 5
        return (dist<=threshold).all()

    def dominatesAll(self, other):
        # used for printing generation summary
        dominates = True
        for index in range(len(self.fitnesses)):
            dominates = dominates and (self.fitnesses[index] > other.fitnesses[index])
        return dominates

    def evaluate(self, gen):
        if self.needs_eval == True:
            self.makeGrid()
            self.makeVXD(gen)
            self.needs_eval = False
        return 0

    def Compute_Fitness(self, gen):
        # read the targets
        with open('targets/cylinder.dat', "rb") as f:
            target1 = np.array(pickle.load(f))
        with open('targets/sphere.dat', "rb") as f:
            target2 = np.array(pickle.load(f))

        # read the results for xml file
        filename = "Data/data{0}/results.xml".format(gen)
        with open(filename, 'r') as f:
            tree = etree.parse(f)

        robots = tree.xpath("//detail/*")
        for robot in robots:
            if robot.tag == 'genome{0}_1'.format(self.ID):
                temp1 = robot.xpath("init_pos")[0].text.replace(';', ',')
                temp2 = np.fromstring(temp1, dtype=float, sep=',')
                positions_i = np.reshape(temp2, (int(np.size(temp2)/3), 3))
                positions_i_z = np.zeros(self.voxelNum)
                for i in range(self.voxelNum):
                    positions_i_z[i] = positions_i[i, 2] + positions_i[i+self.voxelNum, 2]
                    positions_i_z[i] = (positions_i_z[i] + positions_i[i+2*self.voxelNum, 2]) / 3
                
                temp1 = robot.xpath("pos")[0].text.replace(';', ',')
                temp2 = np.fromstring(temp1, dtype=float, sep=',')
                positions1 = np.reshape(temp2, (int(np.size(temp2)/3), 3))
                positions1_z = np.zeros(self.voxelNum)
                for i in range(self.voxelNum):
                    positions1_z[i] = positions1[i, 2] + positions1[i+self.voxelNum, 2]
                    positions1_z[i] = (positions1_z[i] + positions1[i+2*self.voxelNum, 2]) / 3

            elif robot.tag == 'genome{0}_2'.format(self.ID):
                temp1 = robot.xpath("pos")[0].text.replace(';', ',')
                temp2 = np.fromstring(temp1, dtype=float, sep=',')
                positions2 = np.reshape(temp2, (int(np.size(temp2)/3), 3))
                positions2_z = np.zeros(self.voxelNum)
                for i in range(self.voxelNum):
                    positions2_z[i] = positions2[i, 2] + positions2[i+self.voxelNum, 2]
                    positions2_z[i] = (positions2_z[i] + positions2[i+2*self.voxelNum, 2]) / 3
        
        
        o1 = np.sum(np.logical_or(self.top_grid_c1, self.bot_grid_c1))
        o2 = np.sum(np.logical_or(self.top_grid_c2, self.bot_grid_c2))

        self.overlap = (o1 + o2)/(2*c.GRID_SIZE*c.GRID_SIZE)

        # targets are in mm
        heights1 = np.multiply(1000, positions1[0:self.voxelNum, 2]).round(0)
        heights2 = np.multiply(1000, positions2[0:self.voxelNum, 2]).round(0)

        n = 0
        errs = 0
        max_errs = 0
        for i in self.indices: #range(len(target1)): #1941):
            #max_error = abs(positions_i_z[i] - target1[i])
            error = np.abs(np.subtract(heights1[i], target1[i]))
            #if max_error != 0:
            errs = errs + error
            #    max_errs = max_errs + max_error
            n = n + 1
        avg_err1 = errs #(errs / n) / (max_errs / n)
        fitness1 = errs / n

        n = 0
        errs = 0
        max_errs = 0
        for i in self.indices: #range(len(target2)): #1941):
            #max_error = abs(positions_i_z[i] - target2[i])
            error = np.abs(np.subtract(heights2[i], target2[i]))
            #if max_error != 0:
            errs = errs + error
            #    max_errs = max_errs + max_error
            n = n + 1
        avg_err2 = errs #(errs / n) / (max_errs / n)
        fitness2 = errs / n

        self.error1 = avg_err1
        self.error2 = avg_err2

        # we want to minimize error1, error2 and overlap between fibers of two configurations
        self.fitnesses[0] = -1 * round(avg_err1, 1)
        self.fitnesses[1] = -1 * round(avg_err2, 1)
        self.metric = -1 * (1+avg_err1) * (1+avg_err2)

    # make the top and bottom sheets of fiber placements
    def makeGrid(self):
        ## make grids of where the fibers are
        t_points = np.arange(0, 1, 0.01)

        # first configuration - top grid
        self.top_grid_c1 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        for n in range(int(self.genome2[0])):
            points = np.array([[self.genome1[n, 0], self.genome1[n, 1]], [self.genome1[n, 2], self.genome1[n, 3]]])
            curve = Bezier.Curve(t_points, points)
            grid = Bezier.DistCurve(curve)
            self.top_grid_c1 = np.logical_or(self.top_grid_c1, grid)
        # first configuration - bottom grid
        self.bot_grid_c1 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        for n in range(int(self.genome2[1])):
            points = np.array([[self.genome1[c.FIBERS+n, 0], self.genome1[c.FIBERS+n, 1]], [self.genome1[c.FIBERS+n, 2], self.genome1[c.FIBERS+n, 3]]])
            curve = Bezier.Curve(t_points, points)
            grid = Bezier.DistCurve(curve)
            self.bot_grid_c1 = np.logical_or(self.bot_grid_c1, grid)

        # second configuration - top grid
        self.top_grid_c2 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        for n in range(int(self.genome2[2])):
            points = np.array([[self.genome1[int(self.genome2[0])+n, 0], self.genome1[int(self.genome2[0])+n, 1]], [self.genome1[int(self.genome2[0])+n, 2], self.genome1[int(self.genome2[0])+n, 3]]])
            curve = Bezier.Curve(t_points, points)
            grid = Bezier.DistCurve(curve)
            self.top_grid_c2 = np.logical_or(self.top_grid_c2, grid)
        
        # second configuration - bottom grid
        self.bot_grid_c2 = np.zeros((c.GRID_SIZE, c.GRID_SIZE))
        for n in range(int(self.genome2[3])):
            points = np.array([[self.genome1[int(self.genome2[1])+c.FIBERS+n, 0], self.genome1[int(self.genome2[1])+c.FIBERS+n, 1]], [self.genome1[int(self.genome2[1])+c.FIBERS+n, 2], self.genome1[int(self.genome2[1])+c.FIBERS+n, 3]]])
            curve = Bezier.Curve(t_points, points)
            grid = Bezier.DistCurve(curve)
            self.bot_grid_c2 = np.logical_or(self.bot_grid_c2, grid)
    
    # a grid of fibers: 0: no fiber 1: unjammed 2: jammed
    def makeVXD(self, gen):
        # make materials
        DIM = c.DIM
        DIAMETER = c.DIAMETER # in meters

        if not os.path.isdir(f"Data/data{gen}"):
            sub.call("mkdir Data/data"+format(gen), shell=True)

        if not os.path.exists("Data/data{}/base.vxa".format(gen)):
            vxa = VXA(SavePositionOfAllVoxels = 1, EnableExpansion=1, DtFrac = 1, SimTime=0.4, Lattice_Dim = DIAMETER/DIM, BondDampingZ = 1.0, GravEnabled = 0, GravAcc = 0, FloorEnabled = 0, EnableCollision = 0, TempEnabled=1, VaryTempEnabled=1, TempPeriod=1, TempBase=25, TempAmplitude=30)

            RHO = 938.33 #density of the material
            ES = 82700 #moduli of top and bottom passive layers
            EF_j = 1330688 #modulus of jammed fiber
            EF_uj = 68900 #modulus of unjammed fiber
            EE = 160000 #modulus of the expanding layer

            silicone = vxa.add_material(RGBA=(0, 0, 255), E=ES, RHO=RHO) # passive
            expand = vxa.add_material(RGBA=(0, 255, 0), E=EE, RHO=RHO, CTE=0.01) # active
            fiber_j = vxa.add_material(RGBA=(255, 0, 0), E=EF_j, RHO=RHO) #, CTE=-0.001)
            fiber_uj = vxa.add_material(RGBA=(125, 0, 0), E=EF_uj, RHO=RHO)

            vxa.write("Data/data{}/base.vxa".format(gen))

        else:
            vxa = VXA(SavePositionOfAllVoxels = 1, EnableExpansion=1, DtFrac = 1, SimTime=0.4, Lattice_Dim = DIAMETER/DIM, BondDampingZ = 1.0, GravEnabled = 0, GravAcc = 0, FloorEnabled = 0, EnableCollision = 0, TempEnabled=1, VaryTempEnabled=1, TempPeriod=1, TempBase=25, TempAmplitude=30)

            RHO = 938.33 #density of the material
            ES = 82700 #moduli of top and bottom passive layers
            EF_j = 1330688 #modulus of jammed fiber
            EF_uj = 68900 #modulus of unjammed fiber
            EE = 160000 #modulus of the expanding layer

            silicone = vxa.add_material(RGBA=(0, 0, 255), E=ES, RHO=RHO) # passive
            expand = vxa.add_material(RGBA=(0, 255, 0), E=EE, RHO=RHO, CTE=0.01) # active
            fiber_j = vxa.add_material(RGBA=(255, 0, 0), E=EF_j, RHO=RHO) #, CTE=-0.001)
            fiber_uj = vxa.add_material(RGBA=(125, 0, 0), E=EF_uj, RHO=RHO)
            
        # generate robot's body
        center = DIM / 2.0 - 1
        radius = DIM / 2.0

        # to keep track of the number of voxels
        counter = 0
        # first configuration
        body = np.zeros((DIM, DIM, 3))
        for x in range(DIM):
            for y in range(DIM):
                if (math.sqrt(pow(x-center, 2.0)+pow(y-center, 2.0)) < radius):
                    counter = counter + 1
                    body[x, y, 0] = silicone
                    body[x, y, 1] = expand
                    body[x, y, 2] = silicone
                    if self.top_grid_c1[x, y] == 1:
                        body[x, y, 2] = fiber_j
                    elif self.top_grid_c2[x, y] == 1:
                        body[x, y, 2] = fiber_uj
                    if self.bot_grid_c1[x, y] == 1:
                        body[x, y, 0] = fiber_j
                    elif self.bot_grid_c2[x, y] == 1:
                        body[x, y, 0] = fiber_uj

        self.voxelNum = counter
        # generate the VXD file
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1)
        vxd.set_data(body)
        # save the vxd to data folder
        vxd.write("Data/data{0}/genome{1}_1.vxd".format(gen, self.ID))

        # second configuration
        body = np.zeros((DIM, DIM, 3))
        for x in range(DIM):
            for y in range(DIM):
                if (math.sqrt(pow(x-center, 2.0)+pow(y-center, 2.0)) < radius):
                    body[x, y, 0] = silicone
                    body[x, y, 1] = expand
                    body[x, y, 2] = silicone
                    if self.top_grid_c1[x, y] == 1:
                        body[x, y, 2] = fiber_uj
                    elif self.top_grid_c2[x, y] == 1:
                        body[x, y, 2] = fiber_j
                    if self.bot_grid_c1[x, y] == 1:
                        body[x, y, 0] = fiber_uj
                    elif self.bot_grid_c2[x, y] == 1:
                        body[x, y, 0] = fiber_j

        # generate the VXD file
        vxd = VXD()
        vxd.set_tags(RecordVoxel=1)
        vxd.set_data(body)
        # save the vxd to data folder
        vxd.write("Data/data{0}/genome{1}_2.vxd".format(gen, self.ID))

    def mutate(self):
        # sometimes you can't apply one type of mutation, we'll loop through this until we apply some type of mutation
        mutationDone = False
        while not mutationDone:
            type = random.random()
            # type1: move starting or ending point of one fiber
            if type < 0.75:
                top_bot = random.random()
                if top_bot < 0.5: # apply mutation to top sheet
                    if (int(self.genome2[0]+self.genome2[2])>0): # if there are any fibers on the top sheet at all
                        fiber = np.random.randint(0, high=int(self.genome2[0]+self.genome2[2]), dtype='int')
                        grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                        t_points = np.arange(0, 1, 0.01) # for descritization
                        for i in range(int(self.genome2[0]+self.genome2[2])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                            if i != fiber:
                                points_ = np.array([[self.genome1[i, 0], self.genome1[i, 1]], [self.genome1[i, 2], self.genome1[i, 3]]])
                                curve = Bezier.Curve(t_points, points_)
                                temp = Bezier.DistCurve(curve)
                                grid = temp + grid
                        # now let's move the starting/ending point of the chosen fiber - we have to check all three constraints:
                        allCheck = False
                        repeats = 0 # to keep track of how many times we are stuck in the loop. If more than 10, break and try a different mutation
                        while not allCheck:
                            if repeats > 10:
                                break
                            repeats = repeats + 1
                            startEnd = random.random()
                            variation = np.random.randint(0, high=c.GRID_SIZE-1, size=2, dtype='int')
                            candidate = np.array(self.genome1[fiber, :])
                            if startEnd < 0.5: # move the start point
                                candidate[0:2] = variation
                            else: # move the end point
                                candidate[2:4] = variation
                            if np.all(candidate == self.genome1[fiber, :]): # make sure the variation is not zero, meaning that the mutated one must be different than the parent
                                continue
                            # constraint#1: make sure the points have the minimum length
                            check1 = self.checkLength(candidate)
                            # constraint#2: make sure the points are inside the circular sheet
                            check2 = self.checkCircle(candidate)
                            # constraint#3: check the number of intersections
                            points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                            curve = Bezier.Curve(t_points, points_)
                            temp = Bezier.DistCurve(curve)
                            gridCheck = temp + grid
                            check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                            allCheck = np.all([check1, check2, check3])
                        if repeats > 10:
                            mutationDone = False
                        else:
                            # we're all set, save the candidate to the genome
                            self.genome1[fiber, :] = copy.deepcopy(candidate)
                            mutationDone = True
                else: # apply mutation to the bottom sheet
                    if (int(self.genome2[1]+self.genome2[3])>0): # if there are any fibers on the bottom sheet
                        fiber = np.random.randint(0, high=int(self.genome2[1]+self.genome2[3]), dtype='int')
                        grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                        t_points = np.arange(0, 1, 0.01) # for descritization
                        for i in range(int(self.genome2[1]+self.genome2[3])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                            if i != fiber:
                                points_ = np.array([[self.genome1[c.FIBERS+i, 0], self.genome1[c.FIBERS+i, 1]], [self.genome1[c.FIBERS+i, 2], self.genome1[c.FIBERS+i, 3]]])
                                curve = Bezier.Curve(t_points, points_)
                                temp = Bezier.DistCurve(curve)
                                grid = temp + grid
                        # now let's move the starting/ending point of the chosen fiber - we have to check all three constraints:
                        allCheck = False
                        repeats = 0
                        while not allCheck:
                            repeats = repeats + 1
                            if repeats > 10:
                                break
                            startEnd = random.random()
                            variation = np.random.randint(0, high=c.GRID_SIZE-1, size=2, dtype='int')
                            candidate = np.array(self.genome1[c.FIBERS+fiber, :])
                            if startEnd < 0.5: # move the start point
                                candidate[0:2] = variation
                            else: # move the end point
                                candidate[2:4] = variation
                            if np.all(candidate == self.genome1[c.FIBERS+fiber, :]): # make sure the variation is not zero, meaning that the mutated one must be different than the parent
                                continue
                            # constraint#1: make sure the points have the minimum length
                            check1 = self.checkLength(candidate)
                            # constraint#2: make sure the points are inside the circular sheet
                            check2 = self.checkCircle(candidate)
                            # constraint#3: check the number of intersections
                            points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                            curve = Bezier.Curve(t_points, points_)
                            temp = Bezier.DistCurve(curve)
                            gridCheck = temp + grid
                            check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                            allCheck = np.all([check1, check2, check3])
                        if repeats > 10:
                            mutationDone = False
                        else:
                            # we're all set, save the candidate to the genome
                            self.genome1[c.FIBERS+fiber, :] = copy.deepcopy(candidate)
                            mutationDone = True
            # type 2: change the number of fibers on top or bottom in configuration1 or configuration2
            else:
                print("type2")
                done = False
                majorRepeat = 0
                while not done: # make sure something is mutated
                    top_bot = random.random()
                    if top_bot < 0.5: # apply mutation to top sheet
                        c1_c2 = random.random()
                        if c1_c2 < 0.5: # apply mutation to config1
                            add_remove = random.random()
                            if add_remove < 0.5: #add a new fiber
                                if self.genome2[0]+self.genome2[2] < c.FIBERS: # check to see if we can add a new fiber or we are full
                                    grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                                    t_points = np.arange(0, 1, 0.01) # for descritization
                                    for i in range(int(self.genome2[0]+self.genome2[2])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                                        points_ = np.array([[self.genome1[i, 0], self.genome1[i, 1]], [self.genome1[i, 2], self.genome1[i, 3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        grid = temp + grid
                                    # now let's make a new fiber
                                    allCheck = False
                                    repeats = 0
                                    while not allCheck:
                                        repeats = repeats + 1
                                        candidate = np.random.randint(0, high=c.GRID_SIZE, size=4, dtype='int')
                                        # constraint#1: make sure the points have the minimum length
                                        check1 = self.checkLength(candidate)
                                        # constraint#2: make sure the points are inside the circular sheet
                                        check2 = self.checkCircle(candidate)
                                        # constraint#3: check the number of intersections
                                        points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        gridCheck = temp + grid
                                        check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                                        allCheck = np.all([check1, check2, check3])
                                        if repeats > 10:
                                            break
                                    if repeats > 10:
                                        majorRepeat = majorRepeat + 1
                                    else:
                                        # we're all set, save the candidate to the genome
                                        for i in range(int(self.genome2[0]+self.genome2[2]-1), int(self.genome2[0])-1, -1): # need to shift everything down
                                            self.genome1[i+1] = self.genome1[i]
                                        self.genome1[int(self.genome2[0]), :] = copy.deepcopy(candidate)
                                        self.genome2[0] = self.genome2[0]+1
                                        done = True
                            else: # remove a fiber
                                if self.genome2[0] > 0: # check if there are any fibers for config 1 on top sheet
                                    fiber = np.random.randint(0, high=int(self.genome2[0]), size=1, dtype='int')
                                    for i in range(int(fiber), c.FIBERS-1): # shift the fiber positions and remove the entry
                                        self.genome1[i] = self.genome1[i+1]
                                    self.genome2[0] = self.genome2[0] - 1
                                    done = True

                        else: # apply mutation to config2
                            add_remove = random.random()
                            if add_remove < 0.5: #add a new fiber
                                if self.genome2[0]+self.genome2[2] < c.FIBERS: # check to see if we can add a new fiber or we are full
                                    grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                                    t_points = np.arange(0, 1, 0.01) # for descritization
                                    for i in range(int(self.genome2[0]+self.genome2[2])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                                        points_ = np.array([[self.genome1[i, 0], self.genome1[i, 1]], [self.genome1[i, 2], self.genome1[i, 3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        grid = temp + grid
                                    # now let's make a new fiber
                                    allCheck = False
                                    repeats = 0
                                    while not allCheck:
                                        repeats = repeats + 1
                                        candidate = np.random.randint(0, high=c.GRID_SIZE, size=4, dtype='int')
                                        # constraint#1: make sure the points have the minimum length
                                        check1 = self.checkLength(candidate)
                                        # constraint#2: make sure the points are inside the circular sheet
                                        check2 = self.checkCircle(candidate)
                                        # constraint#3: check the number of intersections
                                        points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        gridCheck = temp + grid
                                        check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                                        allCheck = np.all([check1, check2, check3])
                                        if repeats > 10:
                                            break
                                    if repeats > 10:
                                        majorRepeat = majorRepeat + 1
                                    else:
                                        # we're all set, save the candidate to the genome
                                        self.genome1[int(self.genome2[0]+self.genome2[2]), :] = copy.deepcopy(candidate)
                                        self.genome2[2] = self.genome2[2]+1
                                        done = True
                            else: # remove a fiber
                                if self.genome2[2] > 0: # check if there are any fibers for config 2 on top sheet
                                    fiber = np.random.randint(int(self.genome2[0]), high=int(self.genome2[0] + self.genome2[2]), size=1, dtype='int')
                                    for i in range(int(fiber), c.FIBERS-1): # shift the fiber positions and remove the entry
                                        self.genome1[i] = self.genome1[i+1]
                                    self.genome2[2] = self.genome2[2] - 1
                                    done = True
                    else: # apply mutation to bottom sheet
                        c1_c2 = random.random()
                        if c1_c2 < 0.5: # apply mutation to config1
                            add_remove = random.random()
                            if add_remove < 0.5: #add a new fiber
                                if self.genome2[1]+self.genome2[3] < c.FIBERS: # check to see if we can add a new fiber or we are full
                                    grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                                    t_points = np.arange(0, 1, 0.01) # for descritization
                                    for i in range(c.FIBERS, c.FIBERS + int(self.genome2[1]+self.genome2[3])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                                        points_ = np.array([[self.genome1[i, 0], self.genome1[i, 1]], [self.genome1[i, 2], self.genome1[i, 3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        grid = temp + grid
                                    # now let's make a new fiber
                                    allCheck = False
                                    repeats = 0
                                    while not allCheck:
                                        repeats = repeats + 1
                                        candidate = np.random.randint(0, high=c.GRID_SIZE, size=4, dtype='int')
                                        # constraint#1: make sure the points have the minimum length
                                        check1 = self.checkLength(candidate)
                                        # constraint#2: make sure the points are inside the circular sheet
                                        check2 = self.checkCircle(candidate)
                                        # constraint#3: check the number of intersections
                                        points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        gridCheck = temp + grid
                                        check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                                        allCheck = np.all([check1, check2, check3])
                                        if repeats > 10:
                                            break
                                    if repeats > 10:
                                        majorRepeat = majorRepeat + 1
                                    else:
                                        # we're all set, save the candidate to the genome
                                        for i in range(c.FIBERS+int(self.genome2[1]+self.genome2[3])-1, int(c.FIBERS+self.genome2[1]), -1): # need to shift everything down
                                            self.genome1[i+1] = self.genome1[i]
                                        self.genome1[c.FIBERS+int(self.genome2[1]), :] = copy.deepcopy(candidate)
                                        self.genome2[1] = self.genome2[1]+1
                                        done = True
                            else: # remove a fiber
                                if self.genome2[1] > 0: # check if there are any fibers for config 1 on bot sheet
                                    fiber = np.random.randint(c.FIBERS, high=c.FIBERS+int(self.genome2[1]), size=1, dtype='int')
                                    for i in range(int(fiber), 2*c.FIBERS-1): # shift the fiber positions and remove the entry
                                        self.genome1[i] = self.genome1[i+1]
                                    self.genome2[1] = self.genome2[1] - 1
                                    done = True

                        else: # apply mutation to config2
                            add_remove = random.random()
                            if add_remove < 0.5: #add a new fiber
                                if self.genome2[1]+self.genome2[3] < c.FIBERS: # check to see if we can add a new fiber or we are full
                                    grid = np.zeros((c.GRID_SIZE, c.GRID_SIZE)) # to keep track of intersections
                                    t_points = np.arange(0, 1, 0.01) # for descritization
                                    for i in range(c.FIBERS, c.FIBERS+int(self.genome2[1]+self.genome2[3])): # let's make the intersection grid while exclusing the fiber that was chosen to be mutated
                                        points_ = np.array([[self.genome1[i, 0], self.genome1[i, 1]], [self.genome1[i, 2], self.genome1[i, 3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        grid = temp + grid
                                    # now let's make a new fiber
                                    allCheck = False
                                    repeats = 0
                                    while not allCheck:
                                        repeats = repeats + 1
                                        candidate = np.random.randint(0, high=c.GRID_SIZE, size=4, dtype='int')
                                        # constraint#1: make sure the points have the minimum length
                                        check1 = self.checkLength(candidate)
                                        # constraint#2: make sure the points are inside the circular sheet
                                        check2 = self.checkCircle(candidate)
                                        # constraint#3: check the number of intersections
                                        points_ = np.array([[candidate[0], candidate[1]], [candidate[2], candidate[3]]])
                                        curve = Bezier.Curve(t_points, points_)
                                        temp = Bezier.DistCurve(curve)
                                        gridCheck = temp + grid
                                        check3 = np.all(gridCheck<c.MAX_INTERSECT+1)
                                        allCheck = np.all([check1, check2, check3])
                                        if repeats > 10:
                                            break
                                    if repeats > 10:
                                        majorRepeat = majorRepeat + 1
                                    else:
                                        # we're all set, save the candidate to the genome
                                        self.genome1[c.FIBERS+int(self.genome2[1]+self.genome2[3]), :] = copy.deepcopy(candidate)
                                        self.genome2[3] = self.genome2[3]+1
                                        done = True
                            else: # remove a fiber
                                if self.genome2[3] > 0: # check if there are any fibers for config 2 on bot sheet
                                    fiber = np.random.randint(c.FIBERS+int(self.genome2[1]), high=c.FIBERS+int(self.genome2[1] + self.genome2[3])+1, size=1, dtype='int')
                                    for i in range(int(fiber), 2*c.FIBERS-1): # shift the fiber positions and remove the entry
                                        self.genome1[i] = self.genome1[i+1]
                                    self.genome2[3] = self.genome2[3] - 1
                                    done = True
                    if majorRepeat > 10 and not done: # we've been stuck in here for a while and also the mutation is not done, go back to the main loop
                        break
                if majorRepeat < 10: # if mutation is done or if we're just desprete for another type of mutation
                    mutationDone = True
        
        self.needs_eval = True
        self.fitnesses = [0 for x in range(len(self.fitnesses))]

    def get_minimize_vals(self):
        return [self.age]

    def get_maximize_vals(self):
        return self.fitnesses

    def genomePrint(self):
        print(' [fitness: ' , end = '' )
        print(self.fitnesses , end = '' )

        print(' age: ', end = '' )
        print(str(self.age)+']', end = '' )

        print()

        print(self.genome1)
        print(self.genome2)

        print()

    def genomeShow(self):
        print("fitness is: ")
        print(self.fitnesses)

    def SaveBest(self, gen):
        sub.call("mkdir Data/result"+format(gen), shell=True)

        file = open(f'Data/result{gen}/fibers{self.ID}_1.csv', 'w')
        l = np.count_nonzero(self.top_grid_c1) + np.count_nonzero(self.bot_grid_c1)
        file.write(f'{l} ')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                if self.top_grid_c1[i, j]:
                    file.write(f'{i+1} {j+1} -2 ')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                if self.bot_grid_c1[i, j]:
                    file.write(f'{i+1} {j+1} 0 ')
        file.close()

        file = open(f'Data/result{gen}/grid{self.ID}_1.csv', 'w')
        file.write('top grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.top_grid_c1[i, j])} ')
            file.write('\n')

        file.write('\nbot grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.bot_grid_c1[i, j])} ')
            file.write('\n')
        file.close()

        file = open(f'Data/result{gen}/fibers{self.ID}_2.csv', 'w')
        l = np.count_nonzero(self.top_grid_c2) + np.count_nonzero(self.bot_grid_c2)
        file.write(f'{l} ')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                if self.bot_grid_c2[i, j]:
                    file.write(f'{i+1} {j+1} -2 ')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                if self.bot_grid_c2[i, j]:
                    file.write(f'{i+1} {j+1} 0 ')
        file.close()

        file = open(f'Data/result{gen}/grid{self.ID}_2.csv', 'w')
        file.write('top grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.top_grid_c2[i, j])} ')
            file.write('\n')

        file.write('\nbot grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.bot_grid_c2[i, j])} ')
            file.write('\n')
        file.close()

    def SaveForExp(self, num):

        # voxel size in mm
        vxSize = (c.DIAMETER / c.DIM)*1000

        file = open(f'Results{num}/fibers{self.ID}_1.csv', 'w')
        file.write('top grid:\n')
        for n in range(int(self.genome2[0])):
            file.write(f'({self.genome1[n, 0]*vxSize} {self.genome1[n, 1]*vxSize}), ')
            file.write(f'({self.genome1[n, 2]*vxSize} {self.genome1[n, 3]*vxSize}), ')
            file.write(f'({math.sqrt(pow(abs(self.genome1[n, 0]*vxSize-self.genome1[n, 2]*vxSize), 2) + pow(abs(self.genome1[n, 3]*vxSize-self.genome1[n, 1]*vxSize), 2))})')
            file.write('\n')
        
        file.write('bot grid:\n')
        for n in range(int(self.genome2[1])):
            file.write(f'({self.genome1[c.FIBERS+n, 0]*vxSize} {self.genome1[c.FIBERS+n, 1]*vxSize}), ')
            file.write(f'({self.genome1[c.FIBERS+n, 2]*vxSize} {self.genome1[c.FIBERS+n, 3]*vxSize}), ')
            file.write(f'({math.sqrt(pow(abs(self.genome1[c.FIBERS+n, 0]*vxSize-self.genome1[c.FIBERS+n, 2]*vxSize), 2) + pow(abs(self.genome1[c.FIBERS+n, 3]*vxSize-self.genome1[c.FIBERS+n, 1]*vxSize), 2))})')
            file.write('\n')
        file.close()

        file = open(f'Results{num}/grid{self.ID}_1.csv', 'w')
        file.write('top grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.top_grid_c1[i, j])} ')
            file.write('\n')

        file.write('\nbot grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.bot_grid_c1[i, j])} ')
            file.write('\n')
        file.close()

        file = open(f'Results{num}/fibers{self.ID}_2.csv', 'w')
        file.write('top grid:\n')
        for n in range(int(self.genome2[2])):
            file.write(f'({self.genome1[int(self.genome2[0])+n, 0]*vxSize} {self.genome1[int(self.genome2[0])+n, 1]*vxSize}), ')
            file.write(f'({self.genome1[int(self.genome2[0])+n, 2]*vxSize} {self.genome1[int(self.genome2[0])+n, 3]*vxSize}), ')
            file.write(f'({math.sqrt(pow(abs(self.genome1[int(self.genome2[0])+n, 0]*vxSize-self.genome1[int(self.genome2[0])+n, 2]*vxSize), 2) + pow(abs(self.genome1[int(self.genome2[0])+n, 3]*vxSize-self.genome1[int(self.genome2[0])+n, 1]*vxSize), 2))})')
            file.write('\n')
        
        file.write('bot grid:\n')
        for n in range(int(self.genome2[3])):
            file.write(f'({self.genome1[int(self.genome2[1])+c.FIBERS+n, 0]*vxSize} {self.genome1[int(self.genome2[1])+c.FIBERS+n, 1]*vxSize}), ')
            file.write(f'({self.genome1[int(self.genome2[1])+c.FIBERS+n, 2]*vxSize} {self.genome1[int(self.genome2[1])+c.FIBERS+n, 3]*vxSize}), ')
            file.write(f'({math.sqrt(pow(abs(self.genome1[int(self.genome2[1])+c.FIBERS+n, 0]*vxSize-self.genome1[int(self.genome2[1])+c.FIBERS+n, 2]*vxSize), 2) + pow(abs(self.genome1[int(self.genome2[1])+c.FIBERS+n, 3]*vxSize-self.genome1[int(self.genome2[1])+c.FIBERS+n, 1]*vxSize), 2))})')
            file.write('\n')
        file.close()

        file = open(f'Results{num}/grid{self.ID}_2.csv', 'w')
        file.write('top grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.top_grid_c2[i, j])} ')
            file.write('\n')

        file.write('\nbot grid:\n')
        for i in range(c.GRID_SIZE):
            for j in range(c.GRID_SIZE):
                file.write(f'{int(self.bot_grid_c2[i, j])} ')
            file.write('\n')
        file.close()
        
    def checkLength(self, point):
        # checks the minimum length of the fiber
        sx = point[0]
        sy = point[1]
        ex = point[2]
        ey = point[3]
        # check the length of the fiber, return true if it's ok
        length = math.sqrt(pow(sx-ex, 2)+pow(sy-ey, 2))
        if length < c.MIN_L * c.DIM / c.DIAMETER:
            return False
        return True

    def checkCircle(self, point):
        # checks if the fiber has a section outside the circle
        sx = point[0]
        sy = point[1]
        ex = point[2]
        ey = point[3]
        cx = c.DIM / 2.0 - 1 #center of the circle-sheet
        cy = c.DIM / 2.0 - 1
        R = c.DIM / 2.0
        if math.sqrt(pow(sx-cx, 2)+pow(sy-cy, 2)) > R: #starting point outside
            return False
        if math.sqrt(pow(ex-cx, 2)+pow(ey-cy, 2)) > R: #ending point outside
            return False
        return True
