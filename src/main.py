import pickle
import src.solver as solver
from src.config import *


def main(n, objective, timeLimit, epsLimit, **kwargs):

    iterLimit = kwargs.get('iterLimit', 100)
    modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)

    with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
        instance = pickle.load(file)
        solver.main(
            instance, modelname,
            objective=objective,
            timeLimit=timeLimit,
            epsLimit=epsLimit,
            iterLimit=iterLimit
        )


if __name__ == '__main__':
    n = 1430
    objective = 'utilitarian'
    timeLimit = 90
    epsLimit = 0
    iterLimit = 100
    main(n, objective, timeLimit, epsLimit, iterLimit=iterLimit)

