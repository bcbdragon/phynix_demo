'''
python watch_file.py -p1 python genetic.py -d `pwd`
'''
from deap import base, creator
import random
from deap import tools,gp,algorithms
import operator
import math
import numpy
import numpy as np
from functions import (protected_div,
                       mse_eval,
                       generate_training_data,
                       sine_with_freq,
                      )

'''
This is largely boilerplate copied from the DEAP website for GP.  It runs
GP on a series of primitives and functions. 
'''

#############
# CONSTANTS #
#############

# The number of ephemeral constants - random numbers used in trees to supply
# arguments.
NUM_EPHEMERALS=4

# Basic arithmetic primitives.
DEFAULT_PRIMITIVES=[(operator.add,2),
                    (operator.sub,2),
                    (operator.mul,2),
                    (protected_div,2),
                    (operator.neg,1),
                    #(sine_with_freq,2),
                   ] 


#############
# FUNCTIONS #
#############

def init_primitives(otherPrimitives=[],
                    defaultPrimitives=DEFAULT_PRIMITIVES,
                    numEphemerals=NUM_EPHEMERALS,
                   ):
    '''
    Here, we initialize a primitive set, pset, and feed our primitives into
    this.  otherPrimitives is a list of primitives or functions outside 
    the default.
    '''
    primitives=defaultPrimitives+otherPrimitives
    pset=gp.PrimitiveSet("MAIN",1)

    # We must supply the primitive and its parity.
    for (op,parity) in primitives:

        pset.addPrimitive(op,parity)

    # Renaming arguments for display.
    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")

    # Adding ephemeral constants.
    for i in range(numEphemerals):

        ephemeralName=f'eph_{i}'

        pset.addEphemeralConstant(ephemeralName,
                                  lambda: random.uniform(-1, 1))


    return pset


def init_toolbox(pset,trainingData):
    '''
    Here, we initialize our toolbox, which houses the GP session.  We define
    what the fitness function should look like (a miminization), what an 
    individual should look like (a parse tree), and then build our toolbox 
    to run the GP.
    '''
    creator.create("FitnessMin",base.Fitness,weights=(-1.0,))
    creator.create("Individual",gp.PrimitiveTree,fitness=creator.FitnessMin)
    
    toolbox=base.Toolbox()

    # Closure to evaluate Mean Standard Error on the training data (points)
    def eval_symbolic_regression(individual,points):

        f=toolbox.compile(expr=individual)
        xValues=np.array([x for (x,y) in points])
        vectorized_f=np.vectorize(f)
        predictions=vectorized_f(xValues)
        yValues=np.array([y for (x,y) in points])
        diff=(predictions - yValues)**2
        diff=diff.tolist()
        mse=sum(diff)

        return mse,


    # Expressions are generated from the pset.
    toolbox.register("expr",
                     gp.genHalfAndHalf,
                     pset=pset,
                     min_=1,
                     max_=2)
    # Individuals are expressions.
    toolbox.register("individual", 
                     tools.initIterate, 
                     creator.Individual, 
                     toolbox.expr)
    # Population is a list of individuals.
    toolbox.register("population", 
                     tools.initRepeat, 
                     list, 
                     toolbox.individual)
    # Compilation and evaluation for fitness function.
    toolbox.register("compile",
                     gp.compile, 
                     pset=pset)
    toolbox.register("evaluate", 
                     eval_symbolic_regression,
                     points=trainingData
                    )
    # Selection, reproduction, and mutation.
    toolbox.register("select", 
                     tools.selTournament, 
                     tournsize=3)
    toolbox.register("mate", 
                     gp.cxOnePoint)
    toolbox.register("expr_mut", 
                     gp.genFull, 
                     min_=0, 
                     max_=2)
    toolbox.register("mutate", 
                     gp.mutUniform, 
                     expr=toolbox.expr_mut, 
                     pset=pset)


    return toolbox


def evolve(toolbox):
    '''
    This runs the GP and gives us the most successful specimens, at the top of 
    pop.
    '''
    random.seed(318)
    
    pop=toolbox.population(n=300)
    hof=tools.HallOfFame(1)

    statsFit=tools.Statistics(lambda ind: ind.fitness.values)
    statsSize=tools.Statistics(len)
    mstats=tools.MultiStatistics(fitness=statsFit,size=statsSize)
    mstats.register("avg",numpy.mean)
    mstats.register("std",numpy.std)
    mstats.register("min",numpy.min)
    mstats.register("max",numpy.max)

    pop,log=algorithms.eaSimple(pop, toolbox, 0.5, 0.1, 40, stats=mstats,
                                   halloffame=hof, verbose=True)
    return pop,log,hof


def genetic_fit(trainingData,otherPrimitives=[]):
    '''
    Wrapper for initialization of the primitives and the toolbox, runs the GP.
    '''
    pset=init_primitives(otherPrimitives)
    toolbox=init_toolbox(pset,trainingData)
    pop,log,hof=evolve(toolbox)

    return pop,log,hof


if __name__=='__main__':

    trainingData=generate_training_data()
    pop,log,hof=genetic_fit(trainingData)

    print(pop[-1])

 
