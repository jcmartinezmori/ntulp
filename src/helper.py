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


def plot_sequence(modelname, g, lines_df, samples_df, blocking_IterCountStart, blocking_IterCountStartEnd):

    for blocking_IterCount in range(blocking_IterCountStart, blocking_IterCountStartEnd + 1):
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
            x_N, _, _, _, _, S, _ = pickle.load(file)
            print(sorted(S))
        lines_df['width'] = [LINESCALING * x_N[j] / lines_df.iloc[j].length for j in range(len(x_N))]

        for _, data in g.nodes(data=True):
            data['sample_ct'] = 0
        for i, sample in samples_df.iterrows():
            if i in S:
                g.nodes[sample.o_node]['sample_ct'] += 1
                g.nodes[sample.d_node]['sample_ct'] += 1

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
        folium_map.save('{0}/results/figures/sequence/{1}_{2}_{3}.html'.format(RELPATH, FILENAME, modelname, blocking_IterCount))


def plot_map(modelname, g, lines_df, blocking_IterCount):

    if modelname == 'alllines':
        lines_df['width'] = 2
    else:
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
            x_N, _, _, _, _, S, _ = pickle.load(file)
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


def plot_convergence():

    objectives = ['maximin', 'utilitarian']
    blocking_IterLimit = 100
    blocking_TimeLimits = [600, 1200, 1800]
    blocking_EpsLimit = 0

    data = []
    for objective in objectives:
        for blocking_TimeLimit in blocking_TimeLimits:
            modelname = '{0}-{1}-{2}'.format(objective, blocking_TimeLimit, blocking_EpsLimit)
            for blocking_IterCount in range(blocking_IterLimit + 1):
                try:
                    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                            RELPATH, FILENAME, modelname, blocking_IterCount
                    ), 'rb') as file:
                        _, u_N, tt, cutct, eps, _, kappa = pickle.load(file)
                        util_obj = sum(u_N.values())
                        mxmn_obj = min(u_N.values())
                        data.append(
                            (
                                objective, blocking_TimeLimit, blocking_IterCount, tt,
                                cutct, eps, kappa, util_obj, mxmn_obj
                            )
                        )
                except FileNotFoundError:
                    continue

    df = pd.DataFrame(
        data,
        columns=['objective', 'timelimit', 'itct', 'tt', 'cutct', 'eps', 'kappa', 'util_obj', 'mxmn_obj']
    )
    df['util_obj'] /= 3600
    df['tt'] /= (60 * 60)

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    color_map = {600: HEXYELLOW, 1200: HEXBLUE, 1800: HEXVERMILLION}
    marker_map = {600: 'circle', 1200: 'square', 1800: 'diamond'}
    subplot_map = {'maximin': 1, 'utilitarian': 2}

    plots = (
        ('eps', r'$\textrm{Least Objection (}\epsilon\textrm{)}$'),
        ('kappa', r'$\textrm{Basis Condition Number (}\kappa\textrm{)}$'),
        ('cutct', r'$\textrm{Number of Cuts}$'),
        ('tt', r'$\textrm{Elapsed time [hr.]}$'),
        ('util_obj', r'$\textrm{Utilitarian Social Welfare}$'),
        ('mxmn_obj', r'$\textrm{Maximin Social Welfare}$')
    )
    for col, title in plots:
        fig = make_subplots(
            rows=1, cols=2, shared_xaxes=True, shared_yaxes=True,
            subplot_titles=(r'$\textrm{Maximin Objective}$', r'$\textrm{Utilitarian Objective}$')
        )
        for (objective, timelimit), group_df in df.groupby(['objective', 'timelimit']):
            fig.add_trace(
                go.Scatter(
                    x=group_df['itct'],
                    y=group_df[col],
                    mode='lines+markers',
                    name='Time Limit: {0:.0f} min.'.format(timelimit/60),
                    showlegend=True if subplot_map[objective] == 1 else False,
                    line=dict(color=color_map[timelimit], dash='solid'),
                    marker=dict(color=color_map[timelimit], symbol=marker_map[timelimit], size=6)
                ),
                row=1,
                col=subplot_map[objective]
            )
        if col in {'kappa'}:
            fig.update_yaxes(type='log')
        fig.update_yaxes(row=1, col=1, title_text=title, title_font=dict(size=18))
        fig.update_xaxes(row=1, col=1, range=[0, blocking_IterLimit], title_text=r'$\textrm{Number of Rounds}$', title_font=dict(size=18))
        fig.update_xaxes(row=1, col=2, range=[0, blocking_IterLimit], title_text=r'$\textrm{Number of Rounds}$', title_font=dict(size=18))
        for annotation in fig['layout']['annotations']:
            annotation['font'] = {'size': 20}
            annotation['y'] += 0.01
        fig.update_layout(legend=dict(
            orientation="h",
            entrywidth=150,
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5
        ))

        fig.show()



def plot_utilities():

    objectives = ['utilitarian']
    blocking_IterLimit = 100
    blocking_TimeLimits = [1800]
    blocking_EpsLimit = 0

    data = []
    for objective in objectives:
        for blocking_TimeLimit in blocking_TimeLimits:
            modelname = '{0}-{1}-{2}'.format(objective, blocking_TimeLimit, blocking_EpsLimit)
            for blocking_IterCount in range(blocking_IterLimit + 1):
                try:
                    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'rb') as file:
                        _, u_N, _, _ = pickle.load(file)
                        u_N = sorted(u_N.values())
                        data.append(u_N)
                except FileNotFoundError:
                    continue

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = go.Figure()
    conv = []
    for u in data:
        conv.append(sum(u))
        # conv.append(min(u))
    # for u1, u2 in zip(data, data[1:]):
    #     # conv.append(u1[0] - u2[0])
    #     # conv.append(sum(u1) - sum(u2))
    #     conv.append(max(np.absolute(u1_i - u2_i) for u1_i, u2_i in zip(u1, u2)))
    fig.add_trace(
        go.Scatter(
            x=list(range(len(conv))),
            y=conv,
            mode='lines',
            line={'color': HEXBLUE}
            ),
        )
    fig.show()
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
