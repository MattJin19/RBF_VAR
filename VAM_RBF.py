import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.interpolate import Rbf

data = pd.read_csv(f'datasets/treefarms_stats.csv')

resolution = 200

def add_noise(data, noise_level=1e-8):
    np.random.seed(22)
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

frames = []
for i in range(2, 300, 5):
 
    x_raw = add_noise(data['test_acc_all'].values)[0:i]
    y_raw = add_noise(data['test_f1_all'].values)[0:i]
    z_filtered = data['n_leaves'].values[0:i]

    x_min, x_max = min(x_raw), max(x_raw)
    y_min, y_max = min(y_raw), max(y_raw)
    z_min, z_max = min(z_filtered), max(z_filtered)
    rbf = Rbf(x_raw, y_raw, z_filtered, function='linear')

    x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    z_grid = rbf(x_grid, y_grid)

    heatmap = go.Heatmap(
        z=z_grid,
        x=np.linspace(x_min, x_max, resolution),
        y=np.linspace(y_min, y_max, resolution),
        colorscale='RdBu',
        zmin=z_min,
        zmax=z_max,
        colorbar=dict(title='n_leaves', tickvals=np.arange(np.ceil(z_min), np.floor(z_max) + 1, 1)),
        opacity=0.5
    )

    scatter = go.Scatter(
        x=x_raw,
        y=y_raw,
        mode='markers',
        marker=dict(size=8, color='black', opacity=1)
    )

    frames.append(go.Frame(data=[heatmap, scatter]))

fig = make_subplots()
fig.add_trace(frames[0].data[0])
fig.add_trace(frames[0].data[1])

fig.update_layout(
    updatemenus=[dict(
        type="buttons",
        buttons=[dict(label="Play",
                      method="animate",
                      args=[None, {"frame": {"duration": 500, "redraw": True},
                                   "fromcurrent": True}]),
                 dict(label="Pause",
                      method="animate",
                      args=[[None], {"frame": {"duration": 0, "redraw": False},
                                     "mode": "immediate",
                                     "transition": {"duration": 0}}])]
    )],
    xaxis_title="test_acc_all",
    yaxis_title="test_f1_all",
    width = 1000,
    height = 1000
)

fig.frames = frames
fig.show()