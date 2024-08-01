# streamlit run "streamlit_rbf(1.5).py" --server.enableXsrfProtection false

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from scipy.interpolate import Rbf
import streamlit as st

# Function to add noise to data
def add_noise(data, noise_level=1e-8):
    np.random.seed(22)
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise

# Streamlit app
st.markdown(
    """
    <style>
    .custom-title {
        font-size: 24px;
        font-weight: bold;
    }
    </style>
    <div class="custom-title">Interactive Visualization of TreeFarms</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for controls
st.sidebar.title('Controls')

# File uploader for CSV
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = pd.read_csv('treefarms_stats.csv')  # Load default data

# Update button
if st.sidebar.button('Update Data'):
    if uploaded_file is not None:
        st.session_state.data = pd.read_csv(uploaded_file)
        st.sidebar.success('Data updated successfully!')
    else:
        st.sidebar.warning('Please upload a CSV file first.')

# Use the data from session state
data = st.session_state.data

# Define columns available for dropdowns
columns = data.columns

# Dropdowns for selecting axes and heatmap values
x_axis = st.sidebar.selectbox('Select X-axis', columns)
y_axis = st.sidebar.selectbox('Select Y-axis', columns)
heatmap_value = st.sidebar.selectbox('Select Heatmap Value', columns)

# Slider for selecting scatter marker size
marker_size = st.sidebar.slider('Select Scatter Marker Size', 5, 30, 10, 1)

# Slider for selecting the number of points
num_points = st.sidebar.slider('Select Number of Points', 2, 300, 2, 10)

# Slider for selecting resolution
resolution = st.sidebar.slider('Select Resolution', 50, 500, 100, 10)

# Inputs for controlling the colorbar range
colorbar_min = st.sidebar.number_input('Colorbar Min Value', value=float(data[heatmap_value].min()))
colorbar_max = st.sidebar.number_input('Colorbar Max Value', value=float(data[heatmap_value].max()))

# Checkbox for showing/hiding row index annotations
show_annotations = st.sidebar.checkbox('Show Row Index Annotations', value=True)

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
    zmin=colorbar_min,
    zmax=colorbar_max,
    colorbar=dict(title=heatmap_value, tickvals=np.arange(np.ceil(colorbar_min), np.floor(colorbar_max) + 1, 1)),
    opacity=0.5
)

# Create scatter plot with optional annotations
scatter = go.Scatter(
    x=x_raw,
    y=y_raw,
    mode='markers+text' if show_annotations else 'markers',
    marker=dict(size=marker_size, color='black', opacity=1),
    text=[str(idx) for idx in range(num_points)] if show_annotations else None,
    textposition='top center' if show_annotations else None
)

# Create initial figure
fig = make_subplots()
fig.add_trace(heatmap)
fig.add_trace(scatter)

# Set layout
fig.update_layout(
    xaxis_title=x_axis,
    yaxis_title=y_axis,
    width=1600,
    height=600
)

# Render the plot with container width
st.plotly_chart(fig, use_container_width=True)
