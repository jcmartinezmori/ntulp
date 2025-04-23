import folium
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import pickle
from src.config import *


def load():

    g = ox.load_graphml('{0}/results/preprocessed/{1}.graphml'.format(RELPATH, FILENAME))
    g = nx.Graph(g)

    lines_df = pd.read_pickle('{0}/results/preprocessed/lines_df_{1}.pkl'.format(RELPATH, FILENAME))

    return g, lines_df


def plot_map(modelname, g, lines_df, blocking_IterCount):

    if modelname == 'alllines':
        lines_df['width'] = 2
    else:
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
            x_N, u_N, tt = pickle.load(file)
        lines_df['width'] = [LINESCALING * x_N[j] / lines_df.iloc[j].length for j in range(len(x_N))]

    folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
    folium.TileLayer('OpenStreetMap', opacity=1/5).add_to(folium_map)
    for u, data in g.nodes(data=True):
        if data['sample_ct']:
            folium.CircleMarker(
                location=(data['y'], data['x']), color=HEXBLACK, radius=np.log(1 + 1 * data['sample_ct']), weight=0,
                fill=True, fill_opacity=1, tooltip=u
            ).add_to(folium_map)
    for _, line in lines_df.iterrows():
        folium.PolyLine(
            line.coords, color=line.hexcolor, weight=line.width, opacity=1, tooltip=line.name
        ).add_to(folium_map)
    folium_map.save('{0}/results/figures/maps/{1}_{2}_{3}.html'.format(RELPATH, FILENAME, modelname, blocking_IterCount))

