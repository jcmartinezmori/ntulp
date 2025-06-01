import pandas as pd
import pickle
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from src.config import *


def plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit):

    data = []
    for n in ns:
        for objective in objectives:
            modelname = '{0}-{1}-{2}-{3}'.format(n, objective, timeLimit, epsLimit)
            for iterCount in range(iterLimit + 1):
                try:
                    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(
                            RELPATH, FILENAME, modelname, iterCount
                    ), 'rb') as file:
                        _, u_N, tt, cutct, eps, _, kappa = pickle.load(file)
                        util_obj = sum(u_N.values())
                        mxmn_obj = min(u_N.values())
                        data.append(
                            (
                                n, objective, timeLimit, iterCount,
                                tt, cutct, kappa, eps, util_obj, mxmn_obj
                            )
                        )
                except FileNotFoundError:
                    pass

    df = pd.DataFrame(
        data,
        columns=['n', 'objective', 'timelimit', 'iterCount', 'tt', 'cutct', 'kappa', 'eps', 'util_obj', 'mxmn_obj']
    )
    df['util_obj'] /= len(u_N)
    df['tt'] /= (60 * 60)

    color_map = {14: HEXVERMILLION, 42: HEXYELLOW, 132: HEXBLUE, 429: HEXVERMILLION}
    marker_map = {14: 'circle', 42: 'circle', 132: 'square', 429: 'diamond'}
    subplot_map = {objective: idx + 1 for idx, objective in enumerate(objectives)}

    plots = (
        ('tt', r'$\large \textrm{Elapsed time [hr.]}$'),
        ('cutct', r'$\large \textrm{Number of Cuts}$'),
        ('kappa', r'$\large \textrm{Basis Condition Number (}\kappa\textrm{)}$'),
        ('eps', r'$\large \textrm{Least Objection (}\epsilon\textrm{)}$'),
        ('util_obj', r'$\large \textrm{Utilitarian Social Welfare}$'),
        ('mxmn_obj', r'$\large \textrm{Maximin Social Welfare}$')
    )

    for col, title in plots:
        fig = make_subplots(
            rows=1, cols=len(objectives), shared_xaxes=True, shared_yaxes=True,
            subplot_titles=(r'$\Large \textrm{Maximin Service Plan}$', r'$\Large \textrm{Utilitarian Service Plan}$')
        )
        for (objective, n), group_df in df.groupby(['objective', 'n']):
            fig.add_trace(
                go.Scatter(
                    x=group_df['iterCount'],
                    y=group_df[col],
                    mode='lines+markers',
                    name='MIP Timeout: {0:.0f} min.'.format(timeLimit/60),
                    showlegend=True if subplot_map[objective] == 1 else False,
                    line={'color': color_map[n], 'dash': 'solid'},
                    marker={'color': color_map[n], 'symbol': marker_map[n], 'size': 6}
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
                range=[0, iterLimit], title_text=r'$\large \textrm{Number of Rounds}$', title_font={'size': 18}
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
        # fig.write_image('./results/figures/{0}.png'.format(col), width=800, height=600, scale=4)


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


if __name__ == '__main__':

    ns = [14]
    objectives = ['maximin', 'utilitarian']
    timeLimit = 60
    epsLimit = 0
    iterLimit = 100

    plot_convergence(ns, objectives, timeLimit, epsLimit, iterLimit)
