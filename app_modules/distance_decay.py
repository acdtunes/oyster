"""
Distance Decay analysis module
"""
import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import streamlit as st

def exponential_decay(x, a, b):
    """Exponential decay function"""
    return a * np.exp(-b * x)

def create_visualization(conn_matrix, reef_metrics, n_reefs):
    """Create distance decay relationship plot"""
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]
    reef_coords = reef_metrics[['Longitude', 'Latitude']].iloc[:n_reefs].values
    
    # Calculate pairwise distances
    dist_matrix = cdist(reef_coords, reef_coords, metric='euclidean') * 111  # Convert to km
    
    # Get non-zero, non-diagonal elements
    mask = (conn_matrix > 0) & (dist_matrix > 0)
    distances = dist_matrix[mask]
    connectivity = conn_matrix[mask]
    
    # Fit exponential decay
    try:
        popt, _ = curve_fit(exponential_decay, distances, connectivity, p0=[0.05, 0.5])
        x_fit = np.linspace(0, distances.max(), 100)
        y_fit = exponential_decay(x_fit, *popt)
    except:
        popt = [0, 0]
        x_fit = y_fit = []
    
    # Create scatter plot with fit
    fig = go.Figure()
    
    # Data points
    fig.add_trace(go.Scatter(
        x=distances,
        y=connectivity,
        mode='markers',
        marker=dict(
            size=6,
            color=connectivity,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Connectivity", x=1.1)
        ),
        name='Observed',
        hovertemplate='Distance: %{x:.2f} km<br>Connectivity: %{y:.4f}<extra></extra>'
    ))
    
    # Fitted curve
    if len(x_fit) > 0:
        fig.add_trace(go.Scatter(
            x=x_fit,
            y=y_fit,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Fit: {popt[0]:.3f}×exp(-{popt[1]:.2f}×d)'
        ))
    
    fig.update_layout(
        title="Distance Decay of Connectivity",
        xaxis_title="Distance (km)",
        yaxis_title="Connectivity Strength",
        height=500,
        template="plotly_white",
        showlegend=True
    )
    
    # Calculate statistics
    correlation = np.corrcoef(distances, connectivity)[0, 1]
    half_distance = -np.log(0.5) / popt[1] if popt[1] > 0 else np.inf
    
    return fig, correlation, half_distance, distances.max()

def display_statistics(correlation, half_distance, max_distance):
    """Display distance decay statistics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correlation", f"{correlation:.3f}")
    with col2:
        st.metric("50% Decay Distance", f"{half_distance:.2f} km")
    with col3:
        st.metric("Max Dispersal Range", f"{max_distance:.1f} km")

def render_section(conn_matrix, reef_metrics, n_reefs):
    """Render the complete distance decay section"""
    st.header("Distance Decay Analysis")
    st.markdown("""
    Connectivity strength typically decreases with distance between reefs. 
    This relationship helps predict larval exchange patterns.
    """)
    
    fig, correlation, half_distance, max_distance = create_visualization(conn_matrix, reef_metrics, n_reefs)
    st.plotly_chart(fig, use_container_width=True)
    display_statistics(correlation, half_distance, max_distance)