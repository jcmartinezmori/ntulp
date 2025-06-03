import folium
import geopandas as gpd
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.solver as solver
from src.config import *


def preprocess_load():

    g = ox.load_graphml('{0}/results/preprocess/g_{1}.graphml'.format(RELPATH, FILENAME))
    g = nx.Graph(g)
    stops_df = gpd.read_file(
        '{0}/results/preprocess/stops_df_{1}.gpkg'.format(RELPATH, FILENAME), driver='GPKG'
    ).set_index('osmid')
    lines_df = pd.read_pickle('{0}/results/preprocess/lines_df_{1}.pkl'.format(RELPATH, FILENAME))
    trips_df = pd.read_csv('{0}/results/preprocess/trips_df_{1}.csv'.format(RELPATH, FILENAME))

    return g, stops_df, lines_df, trips_df


def blocking_test(n, objective, timeLimit, epsLimit, iterCount):

    modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)

    with open('{0}/results/instances/instance_{1}_{2}.pkl'.format(RELPATH, FILENAME, n), 'rb') as file:
        instance = pickle.load(file)
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
            RELPATH, FILENAME, modelname, iterCount), 'rb'
    ) as file:
        _, u_N, _, _, _, _, _ = pickle.load(file)

    solver.get_blocking(instance, u_N, divPhase=False, MIPFocus=3, TimeLimit=np.inf)
