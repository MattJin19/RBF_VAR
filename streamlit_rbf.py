import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import Rbf
import streamlit as st

# Load the data
data = pd.read_csv('treefarms_stats.csv')

# Define columns available for dropdowns
columns = ['train_acc', 'train_f1', 'test_acc_complete', 'test_f1_complete',
           'test_acc_all', 'test_f1_all', 'n_leaves', 'train_loss']

# Streamlit app
#st.title('Interactive Visualization of TreeFarms')

st.markdown(
    """
    <style>
    .custom-title {
        font-size: 24px; /* 设置字体大小 */
        font-weight: bold; /* 设置字体粗细 */
    }
    </style>
    <div class="custom-title">Interactive Visualization of TreeFarms</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for controls
st.sidebar.title('Controls')

# Dropdowns for selecting axes and heatmap values
x_axis = st.sidebar.selectbox('Select X-axis', columns)
y_axis = st.sidebar.selectbox('Select Y-axis', columns)
heatmap_value = st.sidebar.selectbox('Select Heatmap Value', columns)

# Slider for selecting scatter marker size
marker_size = st.sidebar.slider('Select Scatter Marker Size', 5, 30, 10, 1)

# Slider for selecting the number of points
num_points = st.sidebar.slider('Select Number of Points', 2, 300, 2, 10)

resolution = 100

def add_noise(data, noise_level=1e-8):
    np.random.seed(22)
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

# Extract data based on selected number of points
x_raw = add_noise(data[x_axis].values)[:num_points]
y_raw = add_noise(data[y_axis].values)[:num_points]
z_filtered = data[heatmap_value].values[:num_points]

# RBF interpolation
x_min, x_max = min(x_raw), max(x_raw)
y_min, y_max = min(y_raw), max(y_raw)
z_min, z_max = min(z_filtered), max(z_filtered)
rbf = Rbf(x_raw, y_raw, z_filtered, function='linear')

# Create grid
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
z_grid = rbf(x_grid, y_grid)

# Create heatmap
heatmap = go.Heatmap(
    z=z_grid,
    x=np.linspace(x_min, x_max, resolution),
    y=np.linspace(y_min, y_max, resolution),
    colorscale='RdBu',
    zmin=z_min,
    zmax=z_max,
    colorbar=dict(title=heatmap_value, tickvals=np.arange(np.ceil(z_min), np.floor(z_max) + 1, 1)),
    opacity=0.5
)

# Create scatter plot with annotations
scatter = go.Scatter(
    x=x_raw,
    y=y_raw,
    mode='markers+text',
    marker=dict(size=marker_size, color='black', opacity=1),
    text=[str(idx) for idx in range(num_points)],  # Annotate with row indices
    textposition='top center'
)

# Create initial figure
fig = make_subplots()
fig.add_trace(heatmap)
fig.add_trace(scatter)

# Set layout
fig.update_layout(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    width=1600,  # Set width greater than height
    height=600   # Set height smaller than width
)

# Render the plot
st.plotly_chart(fig, use_container_width=True)
