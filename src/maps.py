import folium
import numpy as np
import pandas as pd
from src.config import *
import src.helper as helper

g, stops_df, lines_df, trips_df = helper.preprocess_load()

trips_df = pd.read_csv('{0}/results/instances/samples_df_{1}_1430.csv'.format(RELPATH, FILENAME))

for _, data in g.nodes(data=True):
    data['trip_ct'] = 0
for _, trip in trips_df.iterrows():
    g.nodes[int(trip.o_node)]['trip_ct'] += 1
    g.nodes[int(trip.d_node)]['trip_ct'] += 1

folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
folium.TileLayer('OpenStreetMap', opacity=1/5).add_to(folium_map)
for u, data in g.nodes(data=True):
    if data['trip_ct']:
        folium.CircleMarker(
            location=(data['y'], data['x']), color=HEXBLACK, radius=np.log(1 + data['trip_ct']), weight=0,
            fill=True, fill_opacity=1, tooltip=u
        ).add_to(folium_map)
# for _, line in lines_df.iterrows():
#     folium.PolyLine(
#         line.coords, color=line.hexcolor, weight=line.width, opacity=1, tooltip=line.name
#     ).add_to(folium_map)
folium_map.save('{0}/results/maps/map_{1}.html'.format(RELPATH, FILENAME))


# def plot_map(modelname, g, lines_df, blocking_IterCount):
#
#     if modelname == 'alllines':
#         lines_df['width'] = 2
#     else:
#         with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
#             x_N, _, _, _, _, S, _ = pickle.load(file)
#         lines_df['width'] = [LINESCALING * x_N[j] / lines_df.iloc[j].length for j in range(len(x_N))]
#
#     folium_map = folium.Map(location=CENTER, zoom_start=11, tiles=None)
#     folium.TileLayer('OpenStreetMap', opacity=1/5).add_to(folium_map)
#     for u, data in g.nodes(data=True):
#         if data['sample_ct']:
#             folium.CircleMarker(
#                 location=(data['y'], data['x']), color=HEXBLACK, radius=np.log(1 + 1 * data['sample_ct']), weight=0,
#                 fill=True, fill_opacity=1, tooltip=u
#             ).add_to(folium_map)
#     for _, line in lines_df.iterrows():
#         folium.PolyLine(
#             line.coords, color=line.hexcolor, weight=line.width, opacity=1, tooltip=line.name
#         ).add_to(folium_map)
#     folium_map.save('{0}/results/figures/maps/{1}_{2}_{3}.html'.format(RELPATH, FILENAME, modelname, blocking_IterCount))
