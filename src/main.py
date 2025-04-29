import pandas as pd
import pickle
import src.helper as helper
import src.solver as solver
from src.config import *

objective = 'maximin'
blocking_IterLimit = 100
blocking_TimeLimit = 600 * 1
blocking_EpsLimit = 0
modelname = '{0}-{1}-{2}'.format(objective, blocking_TimeLimit, blocking_EpsLimit)

solve = True
if solve:
    with open('{0}/results/instances/{1}.pkl'.format(RELPATH, FILENAME), 'rb') as file:
        instance = pickle.load(file)
    N, J, K, A, B, V = instance
    solver.main(
        instance, modelname, objective=objective,
        blocking_TimeLimit=blocking_TimeLimit,
        blocking_IterLimit=blocking_IterLimit,
        blocking_EpsLimit=blocking_EpsLimit
    )

plot_map = False
if plot_map:
    g, lines_df = helper.load()
    samples_df = pd.read_csv('{0}/results/instances/samples_df_{1}.csv'.format(RELPATH, FILENAME))

    for _, data in g.nodes(data=True):
        data['sample_ct'] = 0
    for _, sample in samples_df.iterrows():
        g.nodes[sample.o_node]['sample_ct'] += 1
        g.nodes[sample.d_node]['sample_ct'] += 1

    helper.plot_map(modelname, g, lines_df, -1)
    helper.plot_map(modelname, g, lines_df, 0)
    # helper.plot_map(modelname, g, lines_df, 2)
    helper.plot_map(modelname, g, lines_df, 90)


plot_util_curves = False
if plot_util_curves:
    pass



