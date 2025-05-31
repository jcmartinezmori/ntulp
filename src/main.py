import pandas as pd
import pickle
import src.helper as helper
import src.solver as solver
from src.config import *

n = 132
objective = 'maximin'
iterLimit = 100
timeLimit = 60
epsLimit = 0
modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)

solve = True
if solve:
    with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
        instance = pickle.load(file)
    N, J, K, A, B, V = instance
    solver.main(
        instance, modelname, objective=objective,
        iterLimit=iterLimit,
        timeLimit=timeLimit,
        epsLimit=epsLimit
    )
