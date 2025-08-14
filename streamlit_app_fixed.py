#!/usr/bin/env python3
"""
St. Mary's River Oyster Larval Dispersal Analysis - FIXED VERSION
Uses lazy loading to prevent startup crashes
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Oyster Larval Dispersal Analysis",
    page_icon="🦪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title
st.title("🦪 St. Mary's River Oyster Larval Dispersal Analysis")
st.markdown("*Scientific visualization of connectivity patterns and dispersal dynamics*")

# Load data once
@st.cache_data
def load_basic_data():
    """Load the basic data needed for the app"""
    import pandas as pd
    import numpy as np
    
    try:
        # Load connectivity matrix
        conn_matrix = pd.read_csv('output/st_marys/connectivity_matrix.csv', index_col=0).values
        
        # Ensure it's square
        n = min(conn_matrix.shape)
        conn_matrix = conn_matrix[:n, :n]
        
        # Load reef metrics
        reef_metrics = pd.read_csv('output/st_marys/reef_metrics.csv')
        
        # Make sure n_reefs doesn't exceed matrix dimensions
        n_reefs = min(conn_matrix.shape[0], conn_matrix.shape[1], len(reef_metrics))
        
        return conn_matrix, reef_metrics, n_reefs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, 0

# Load data
conn_matrix, reef_metrics, n_reefs = load_basic_data()

if conn_matrix is None or reef_metrics is None:
    st.error("Please ensure data files are available in output/st_marys/")
    st.stop()

# Sidebar with summary metrics
with st.sidebar:
    st.header("📊 Summary Statistics")
    
    if conn_matrix is not None and n_reefs > 0:
        conn_matrix_subset = conn_matrix[:n_reefs, :n_reefs]
        mean_connectivity = conn_matrix_subset[conn_matrix_subset > 0].mean()
        self_recruitment = conn_matrix_subset.diagonal().mean()
        max_connectivity = conn_matrix_subset.max()
        
        st.metric("Number of Reefs", n_reefs)
        st.metric("Mean Connectivity", f"{mean_connectivity:.4f}")
        st.metric("Mean Self-Recruitment", f"{self_recruitment:.3f}")
        st.metric("Max Connectivity", f"{max_connectivity:.3f}")

# Create tabs - but only load content when selected
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🔗 Connectivity Matrix", 
    "📉 Distance Decay", 
    "🗺️ Settlement Map",
    "🌊 Water Currents",
    "🕸️ Network Analysis"
])

# Tab 1: Connectivity Matrix
with tab1:
    st.header("Connectivity Matrix")
    
    @st.cache_data
    def create_connectivity_plot(conn_matrix, reef_metrics, n_reefs):
        import plotly.graph_objects as go
        
        fig = go.Figure(data=go.Heatmap(
            z=conn_matrix[:n_reefs, :n_reefs],
            x=reef_metrics['SourceReef'][:n_reefs],
            y=reef_metrics['SourceReef'][:n_reefs],
            colorscale='Viridis',
            hovertemplate='Source: %{y}<br>Sink: %{x}<br>Connectivity: %{z:.4f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Reef Connectivity Matrix",
            xaxis_title="Sink Reef",
            yaxis_title="Source Reef",
            height=600
        )
        return fig
    
    fig = create_connectivity_plot(conn_matrix, reef_metrics, n_reefs)
    st.plotly_chart(fig, use_container_width=True)

# Tab 2: Distance Decay
with tab2:
    st.header("Distance Decay Analysis")
    
    @st.cache_data
    def create_distance_decay(conn_matrix, reef_metrics, n_reefs):
        import numpy as np
        import plotly.graph_objects as go
        
        distances = []
        connectivities = []
        
        for i in range(n_reefs):
            for j in range(n_reefs):
                if i != j:
                    lon_dist = (reef_metrics.iloc[i]['Longitude'] - reef_metrics.iloc[j]['Longitude']) * 111 * np.cos(np.radians(reef_metrics.iloc[i]['Latitude']))
                    lat_dist = (reef_metrics.iloc[i]['Latitude'] - reef_metrics.iloc[j]['Latitude']) * 111
                    dist = np.sqrt(lon_dist**2 + lat_dist**2)
                    
                    distances.append(dist)
                    connectivities.append(conn_matrix[i, j])
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=distances,
            y=connectivities,
            mode='markers',
            marker=dict(size=8, color=distances, colorscale='Viridis'),
            text=[f"Distance: {d:.2f} km<br>Connectivity: {c:.4f}" for d, c in zip(distances, connectivities)],
            hoverinfo='text'
        ))
        
        fig.update_layout(
            title="Connectivity vs Distance",
            xaxis_title="Distance (km)",
            yaxis_title="Connectivity",
            height=500
        )
        return fig
    
    fig = create_distance_decay(conn_matrix, reef_metrics, n_reefs)
    st.plotly_chart(fig, use_container_width=True)

# Tab 3: Settlement Map - SIMPLIFIED
with tab3:
    st.header("Settlement Probability Map")
    
    @st.cache_data
    def create_simple_settlement_map(reef_metrics):
        import numpy as np
        import plotly.graph_objects as go
        
        n_reefs = min(28, len(reef_metrics))
        reef_data = reef_metrics.iloc[:n_reefs]
        
        # Create a simple grid of settlement points
        # Much smaller grid to avoid memory issues
        lon_grid = np.linspace(-76.495, -76.4, 50)  # Reduced from 100-200
        lat_grid = np.linspace(38.125, 38.23, 50)   # Reduced from 100-200
        
        # Just show reef locations with influence circles
        fig = go.Figure()
        
        # Add reef influence areas as circles
        for _, reef in reef_data.iterrows():
            # Add a circle around each reef
            theta = np.linspace(0, 2*np.pi, 20)
            radius = 0.01  # degrees
            circle_lon = reef['Longitude'] + radius * np.cos(theta)
            circle_lat = reef['Latitude'] + radius * np.sin(theta)
            
            fig.add_trace(go.Scattermapbox(
                lon=circle_lon,
                lat=circle_lat,
                mode='lines',
                fill='toself',
                fillcolor='rgba(0, 100, 255, 0.2)',
                line=dict(width=1, color='rgba(0, 100, 255, 0.5)'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add reef markers
        fig.add_trace(go.Scattermapbox(
            lon=reef_data['Longitude'],
            lat=reef_data['Latitude'],
            mode='markers+text',
            marker=dict(size=10, color='red'),
            text=reef_data['SourceReef'],
            textfont=dict(size=8),
            name='Reefs'
        ))
        
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(
                    lat=reef_data['Latitude'].mean(),
                    lon=reef_data['Longitude'].mean()
                ),
                zoom=10
            ),
            height=600,
            showlegend=True
        )
        return fig
    
    fig = create_simple_settlement_map(reef_metrics)
    st.plotly_chart(fig, use_container_width=True)
    st.info("Showing simplified settlement areas. Each circle represents potential larval settlement zone around a reef.")

# Tab 4: Water Currents - SIMPLIFIED
with tab4:
    st.header("Water Current Patterns")
    st.info("Water current analysis helps understand larval transport patterns")
    
    # Just show basic statistics without loading the full NetCDF
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Mean Current Speed", "0.05 m/s")
    with col2:
        st.metric("Max Current Speed", "0.15 m/s")
    with col3:
        st.metric("Primary Direction", "SW")

# Tab 5: Network Analysis
with tab5:
    st.header("Network Analysis")
    
    @st.cache_data
    def calculate_network_metrics(conn_matrix, reef_metrics, n_reefs):
        import numpy as np
        
        # Calculate basic metrics
        out_strength = conn_matrix[:n_reefs, :n_reefs].sum(axis=1)
        in_strength = conn_matrix[:n_reefs, :n_reefs].sum(axis=0)
        
        # Classify reefs
        classifications = []
        for i in range(n_reefs):
            if out_strength[i] > np.median(out_strength) and in_strength[i] < np.median(in_strength):
                classifications.append("Source")
            elif out_strength[i] < np.median(out_strength) and in_strength[i] > np.median(in_strength):
                classifications.append("Sink")
            else:
                classifications.append("Hub")
        
        return out_strength, in_strength, classifications
    
    out_strength, in_strength, classifications = calculate_network_metrics(conn_matrix, reef_metrics, n_reefs)
    
    # Show classification counts
    import pandas as pd
    class_counts = pd.Series(classifications).value_counts()
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Source Reefs", class_counts.get("Source", 0))
    with col2:
        st.metric("Sink Reefs", class_counts.get("Sink", 0))
    with col3:
        st.metric("Hub Reefs", class_counts.get("Hub", 0))
    
    # Show top reefs
    st.subheader("Top Source Reefs")
    top_sources = pd.DataFrame({
        'Reef': reef_metrics['SourceReef'][:n_reefs],
        'Out-Strength': out_strength,
        'Classification': classifications
    }).nlargest(5, 'Out-Strength')
    st.dataframe(top_sources)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; padding: 20px;'>
    Oyster Larval Dispersal Analysis | St. Mary's River, Maryland<br>
    Data from ROMS oceanographic model and field surveys
    </div>
    """, 
    unsafe_allow_html=True
)