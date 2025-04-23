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
            x_N, _, _, _ = pickle.load(file)
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


# def plot_util_curves(modelname1, modelname2):
#
#     with open('{0}/results/solutions/out_{1}_{2}.pkl'.format(relpath, filename, modelname1), 'rb') as file:
#         _, u_N1, _, _, _ = pickle.load(file)
#     with open('{0}/results/solutions/out_{1}_{2}.pkl'.format(relpath, filename, modelname2), 'rb') as file:
#         _, u_N2, _, _, _ = pickle.load(file)
#     n = len(u_N1)
#
#     u1 = sorted(u_N1.values())
#     u2 = sorted(u_N2.values())
#     fig = make_subplots(rows=1, cols=2)
#     fig.add_trace(
#         go.Scatter(
#             x=list(range(n)), y=u1, mode='lines+markers', name=r'$\Large \text{Without cooperation}$', showlegend=True,
#             marker=dict(symbol='square', color=hexblue, size=5), line=dict(color=hexblue, width=1)
#         ), row=1, col=1,
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=list(range(n)), y=u2, mode='lines+markers', name=r'$\Large \text{With cooperation}$', showlegend=True,
#             marker=dict(symbol='diamond', color=hexvermillion, size=5), line=dict(color=hexvermillion, width=1)
#         ), row=1, col=1
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=list(range(n)), y=u1, mode='lines+markers', name=r'$\Large \text{Without cooperation}$', showlegend=False,
#             marker=dict(symbol='square', color=hexblue, size=5), line=dict(color=hexblue, width=1)
#         ), row=1, col=2
#     )
#     fig.add_trace(
#         go.Scatter(
#             x=list(range(n)), y=u2, mode='lines+markers', name=r'$\Large \text{With cooperation}$', showlegend=False,
#             marker=dict(symbol='diamond', color=hexvermillion, size=5), line=dict(color=hexvermillion, width=1)
#         ), row=1, col=2
#     )
#     fig.update_layout(
#         xaxis=dict(
#             title=r'$\Large \text{Riders (Sorted by Utility)}$',
#             title_font=dict(size=30),
#             tickfont=dict(size=16)
#         ),
#         xaxis2=dict(
#             title=r'$\Large \text{Riders (Sorted by Utility)}$',
#             title_font=dict(size=30),
#             tickfont=dict(size=16),
#             range=[0, 360]
#         ),
#         yaxis=dict(
#             title=r'$\Large \text{Utility}$',
#             title_font=dict(size=30),
#             tickfont=dict(size=16)
#         ),
#         yaxis2=dict(
#             tickfont=dict(size=16),
#             range=[0 - 0.0125, 0.3 + 0.0125]
#         ),
#         legend=dict(
#             font=dict(size=30),
#             orientation='h',
#             xanchor='center',
#             yanchor='bottom',
#             x=0.5,
#             y=1.025
#         )
#     )
#     fig.write_html('./results/figures/utilities.html'.format(filename))
#     fig.write_image('./results/figures/utilities.png'.format(filename), width=1600, height=800, scale=3)
