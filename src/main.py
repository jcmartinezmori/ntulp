import pandas as pd
import pickle
import src.solver as solver
from src.config import *

objective = 'maximin'
blocking_TimeLimit = 30
blocking_IterLimit = 3
blocking_EpsLimit = 0
modelname = '{0}-{1}-{2}-{3}'.format(objective, blocking_TimeLimit, blocking_IterLimit, blocking_EpsLimit)

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

# plot = True
# if plot:
#     g, _, lines_df, _ = helper.load()
#     samples_df = pd.read_csv('{0}/results/instance/samples_df_{1}.csv'.format(relpath, filename))
#     for _, data in g.nodes(data=True):
#         data['sample_ct'] = 0
#     for _, sample in samples_df.iterrows():
#         g.nodes[sample.o_node]['sample_ct'] += 1
#         g.nodes[sample.d_node]['sample_ct'] += 1
#
#     helper.plot_map(modelname, g, lines_df)

