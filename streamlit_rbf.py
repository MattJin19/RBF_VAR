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
        margin-bottom: 0px;
    }
    </style>
    <div class="custom-title">Interactive Visualization of TreeFarms</div>
    """,
    unsafe_allow_html=True
)

# Sidebar for controls =================
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
x_col, y_col= st.sidebar.columns(2)
x_axis = x_col.selectbox('Select X-axis', columns)
y_axis = y_col.selectbox('Select Y-axis', columns)
heatmap_value = st.sidebar.selectbox('Select Heatmap Value', columns)

# Slider for selecting scatter marker size
marker_size = st.sidebar.slider('Select Scatter Marker Size', 5, 30, 10, 1)

# Slider for selecting the number of points
num_points = st.sidebar.slider('Select Number of Points', 2, 300, 2, 10)

# Slider for selecting resolution
resolution = st.sidebar.slider('Select Resolution', 50, 500, 100, 10)

# Color pickers for selecting colorbar start and end colors in the same row
col1, col2 = st.sidebar.columns(2)
color_start = col1.color_picker('Start Color', '#FF0000')  # Red
color_end = col2.color_picker('End Color', '#0000FF')  # Blue

# Inputs for controlling the colorbar range
col_min, col_max= st.sidebar.columns(2)
colorbar_min = col_min.number_input('Colorbar Min Value', value=float(data[heatmap_value].min()))
colorbar_max = col_max.number_input('Colorbar Max Value', value=float(data[heatmap_value].max()))

# Checkbox for showing/hiding row index annotations
show_annotations = st.sidebar.checkbox('Show Row Index Annotations', value=True)

# Dropdown to select the RBF function
rbf_function = st.sidebar.selectbox(
    'Select RBF Function',
    ['linear', 'gaussian', 'multiquadric', 'inverse', 'thin_plate', 'cubic']
)

# Add a checkbox to toggle the heatmap
show_heatmap = st.sidebar.checkbox('Show Heatmap', value=True)

# =================

# Extract data based on selected number of points
x_raw = add_noise(data[x_axis].values)[:num_points]
y_raw = add_noise(data[y_axis].values)[:num_points]
z_filtered = data[heatmap_value].values[:num_points]

# RBF interpolation
x_min, x_max = min(x_raw), max(x_raw)
y_min, y_max = min(y_raw), max(y_raw)
z_min, z_max = min(z_filtered), max(z_filtered)
rbf = Rbf(x_raw, y_raw, z_filtered, function=rbf_function)

# Create grid
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
z_grid = rbf(x_grid, y_grid)

# Calculate colors for scatter points
scatter_colors = rbf(x_raw, y_raw)

# Create custom color scale
midpoint_value = (colorbar_min + colorbar_max) / 2
colorscale = [
    [0, color_start],  # Start color
    [(midpoint_value - colorbar_min) / (colorbar_max - colorbar_min), '#FFFFFF'],  # Midpoint as white
    [1, color_end]  # End color
]

# Create heatmap with the new colorscale
heatmap = go.Heatmap(
    z = z_grid,
    x = np.linspace(x_min, x_max, resolution),
    y = np.linspace(y_min, y_max, resolution),
    colorscale = colorscale,
    zmin = colorbar_min,
    zmax = colorbar_max,
    colorbar = dict(title=heatmap_value, tickvals=np.arange(np.ceil(colorbar_min), np.floor(colorbar_max) + 1, 1)),
    opacity = 0.5
)

# Create scatter plot with colors based on RBF values
scatter = go.Scatter(
    x = x_raw,
    y = y_raw,
    mode = 'markers+text' if show_annotations else 'markers',
    marker = dict(
        size=marker_size,
        color=scatter_colors,
        colorscale=colorscale,
        cmin=colorbar_min,
        cmax=colorbar_max,
        opacity=1,
        showscale=not show_heatmap  # Show colorbar for scatter when heatmap is off
    ),
    text = [str(idx) for idx in range(num_points)] if show_annotations else None,
    textposition = 'top center' if show_annotations else None
)

# Create the figure
fig = make_subplots()

# Add traces based on the checkbox state
if show_heatmap:
    fig.add_trace(heatmap)
fig.add_trace(scatter)

# Update layout
fig.update_layout(
    xaxis_title=dict(text=x_axis, font=dict(size=30)),
    yaxis_title=dict(text=y_axis, font=dict(size=30)),
    xaxis=dict(tickfont=dict(size=16)),
    yaxis=dict(tickfont=dict(size=16)),
    width=1600,
    height=600,
    margin=dict(t=20)
)

# Render the plot with container width
st.plotly_chart(fig, use_container_width=True)
