import folium
import osmnx as ox
import networkx as nx
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import src.solver as solver
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
    blocking_TimeLimits = [300, 600, 900]
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
                                objective, blocking_TimeLimit, blocking_IterCount,
                                tt, cutct, eps, kappa, util_obj, mxmn_obj
                            )
                        )
                except FileNotFoundError:
                    pass

    df = pd.DataFrame(
        data,
        columns=['objective', 'timelimit', 'itct', 'tt', 'cutct', 'eps', 'kappa', 'util_obj', 'mxmn_obj']
    )
    df['util_obj'] /= len(u_N)
    df['tt'] /= (60 * 60)

    color_map = {300: HEXYELLOW, 600: HEXBLUE, 900: HEXVERMILLION}
    marker_map = {300: 'circle', 600: 'square', 900: 'diamond'}
    subplot_map = {objective: idx + 1 for idx, objective in enumerate(objectives)}

    plots = (
        ('eps', r'$\large \textrm{Least Objection (}\epsilon\textrm{)}$'),
        ('kappa', r'$\large \textrm{Basis Condition Number (}\kappa\textrm{)}$'),
        ('cutct', r'$\large \textrm{Number of Cuts}$'),
        ('tt', r'$\large \textrm{Elapsed time [hr.]}$'),
        ('util_obj', r'$\large \textrm{Utilitarian Social Welfare}$'),
        ('mxmn_obj', r'$\large \textrm{Maximin Social Welfare}$')
    )

    for col, title in plots:
        fig = make_subplots(
            rows=1, cols=len(objectives), shared_xaxes=True, shared_yaxes=True,
            subplot_titles=(r'$\Large \textrm{Maximin Service Plan}$', r'$\Large \textrm{Utilitarian Service Plan}$')
        )
        for (objective, timelimit), group_df in df.groupby(['objective', 'timelimit']):
            fig.add_trace(
                go.Scatter(
                    x=group_df['itct'],
                    y=group_df[col],
                    mode='lines+markers',
                    name='MIP Timeout: {0:.0f} min.'.format(timelimit/60),
                    showlegend=True if subplot_map[objective] == 1 else False,
                    line={'color': color_map[timelimit], 'dash': 'solid'},
                    marker={'color': color_map[timelimit], 'symbol': marker_map[timelimit], 'size': 6}
                ),
                row=1,
                col=subplot_map[objective]
            )
        if col in {'kappa'}:
            fig.update_yaxes(type='log')
        fig.update_yaxes(
            row=1, col=1,
            title_text=title, title_font={'size': 18}
        )
        for idx, _ in enumerate(objectives):
            fig.update_xaxes(
                row=1, col=idx + 1,
                range=[0, blocking_IterLimit], title_text=r'$\large \textrm{Number of Rounds}$', title_font={'size': 18}
            )
        for annotation in fig['layout']['annotations']:
            annotation['font'] = {'size': 22}
            annotation['y'] = 1.0125
        fig.update_layout(
            legend={
                'orientation': 'h', 'entrywidth': 150, 'yanchor': 'top', 'y': -0.125, 'xanchor': 'right', 'x': 1,
                'font': {'size': 14}
            }
        )
        fig.show()
        fig.write_image('./results/figures/{0}.png'.format(col), width=800, height=600, scale=4)


def plot_utilities():

    objective = 'maximin'
    blocking_TimeLimits = [300]
    blocking_EpsLimit = 0
    blocking_IterCount = 58

    data = []
    for blocking_TimeLimit in blocking_TimeLimits:
        modelname = '{0}-{1}-{2}'.format(objective, blocking_TimeLimit, blocking_EpsLimit)
        try:
            with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                    RELPATH, FILENAME, modelname, blocking_IterCount
            ), 'rb') as file:
                _, u_N, _, _, _, _, _ = pickle.load(file)
                u_N = sorted(u_N.values())
                for idx, u_i in enumerate(u_N):
                    data.append(
                        (
                            blocking_TimeLimit, idx + 1, u_i
                        )
                    )
        except FileNotFoundError:
            pass
    modelname = '{0}-{1}-{2}'.format(objective, blocking_TimeLimits[0], blocking_EpsLimit)
    try:
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                RELPATH, FILENAME, modelname, -1
        ), 'rb') as file:
            _, u_N, _, _, _, _, _ = pickle.load(file)
            u_N = sorted(u_N.values())
            for idx, u_i in enumerate(u_N):
                data.append(
                    (
                        -1, idx + 1, u_i
                    )
                )
    except FileNotFoundError:
        pass

    df = pd.DataFrame(
        data,
        columns=['timelimit', 'idx', 'u_i']
    )

    color_map = {-1: HEXBLACK, 300: HEXYELLOW, 600: HEXBLUE, 900: HEXVERMILLION}
    marker_map = {-1: 'star', 300: 'circle', 600: 'square', 900: 'diamond'}
    dash_map = {-1: 'dot', 300: 'solid', 600: 'solid', 900: 'solid'}

    fig = make_subplots(
        rows=1, cols=1,
        subplot_titles=(r'$\Large \textrm{Maximin Service Plan}$',)
    )

    for (timelimit, ), group_df in df.groupby(['timelimit']):
        fig.add_trace(
            go.Scatter(
                x=group_df['idx'],
                y=group_df['u_i'],
                mode='lines+markers',
                name='MIP Timeout: {0:.0f} min.'.format(timelimit/60) if timelimit != -1 else 'Without Cooperation',
                line={'color': color_map[timelimit], 'dash': dash_map[timelimit]},
                marker={'color': color_map[timelimit], 'symbol': marker_map[timelimit], 'size': 2}
            ),
            row=1, col=1
        )

    fig.update_yaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Utility}$', title_font={'size': 18}
    )
    fig.update_xaxes(
        row=1, col=1,
        title_text=r'$\large \textrm{Riders (Sorted by Utility)}$', title_font={'size': 18}
    )
    for annotation in fig['layout']['annotations']:
        annotation['font'] = {'size': 22}
        annotation['y'] = 1.0125
    fig.update_layout(
        legend={
            'orientation': 'h', 'entrywidth': 150, 'yanchor': 'top', 'y': -0.125, 'xanchor': 'right', 'x': 1,
            'font': {'size': 14}
        }
    )

    fig.show()
    fig.write_image('./results/figures/utils.png', width=800, height=600, scale=4)


def test():

    objective = 'utilitarian'
    timeLimit = 300
    epsLimit = 0
    blocking_IterCount = 69

    with open('{0}/results/instances/{1}.pkl'.format(RELPATH, FILENAME), 'rb') as file:
        instance = pickle.load(file)
    modelname = '{0}-{1}-{2}'.format(objective, timeLimit, epsLimit)
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
            RELPATH, FILENAME, modelname, blocking_IterCount
    ), 'rb') as file:
        _, u_N, _, _, _, _, _ = pickle.load(file)

    solver.get_blocking(instance, u_N, divPhase=False, MIPFocus=3, timeLimit=np.Infinity)
