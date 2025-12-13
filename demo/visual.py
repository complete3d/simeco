import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def get_point_cloud_plot(path):
    """
    Generates a 3D scatter plot trace for a point cloud.

    Parameters:
        title (str): The title of the point cloud plot.
        path (str): Path to the point cloud file (in .npy format).
    """
    # Load point cloud data
    point_cloud_array = np.load(path)
    x = point_cloud_array[:, 0]
    y = point_cloud_array[:, 1]
    z = point_cloud_array[:, 2]
    
    # Return the point cloud trace
    trace = go.Scatter3d(
        x=x, y=y, z=z,
        mode='markers',
        marker=dict(
            size=1,
            color='rgb(245, 222, 179)',
            colorscale='Viridis',
            opacity=0.8,
        ),
        hoverinfo='none'
    )
    return trace


def visualize_point_cloud(pc_title, pc_path, pred_title, pred_path, eye = dict(x=0, y=1.5, z=2)):
    # Create subplot layout with reduced spacing
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'mesh3d'}]],
        column_titles=[pc_title, pred_title],
        horizontal_spacing=0.02,
    )

    # Add point cloud plot
    pc_trace = get_point_cloud_plot(pc_path)
    fig.add_trace(pc_trace, row=1, col=1)
    
    # Add point cloud plot
    pred_trace = get_point_cloud_plot(pred_path)
    fig.add_trace(pred_trace, row=1, col=2)


    # Update layout for both subplots
    fig.update_layout(
        margin=dict(t=20, b=20, l=10, r=10),  
        scene=dict(  
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            zaxis=dict(visible=False, range=[-1, 1]),
            aspectmode="cube",
            camera=dict(
            eye=eye,  
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=0))),
        scene2=dict(  
            xaxis=dict(visible=False, range=[-1, 1]),
            yaxis=dict(visible=False, range=[-1, 1]),
            zaxis=dict(visible=False, range=[-1, 1]),
            aspectmode="cube",
            camera=dict(
            eye=eye,  
            up=dict(x=0, y=0, z=1), 
            center=dict(x=0, y=0, z=0))),
        showlegend=False
    )
        
    fig.show()