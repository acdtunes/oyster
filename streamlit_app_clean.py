#!/usr/bin/env python3
"""
St. Mary's River Oyster Larval Dispersal Analysis
A clean, scientific visualization of connectivity patterns and dispersal dynamics
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import netCDF4 as nc
from scipy.spatial.distance import cdist
from scipy.optimize import curve_fit
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Oyster Larval Dispersal Analysis",
    page_icon="🦪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #cccccc;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load connectivity matrix and reef metrics"""
    try:
        conn_df = pd.read_csv("output/st_marys/connectivity_matrix.csv", index_col=0)
        reef_metrics = pd.read_csv("output/st_marys/reef_metrics.csv")
        
        # Ensure the matrix is square
        n_reefs = min(conn_df.shape[0], conn_df.shape[1], len(reef_metrics))
        
        # Create a square matrix
        conn_matrix = np.zeros((n_reefs, n_reefs))
        for i in range(min(n_reefs, conn_df.shape[0])):
            for j in range(min(n_reefs, conn_df.shape[1])):
                conn_matrix[i, j] = conn_df.iloc[i, j]
        
        return conn_matrix, reef_metrics
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

@st.cache_data
def load_netcdf_data():
    """Load NetCDF oceanographic data"""
    try:
        with nc.Dataset('data/109516.nc', 'r') as dataset:
            lon = dataset.variables['longitude'][:]
            lat = dataset.variables['latitude'][:]
            u_surface = dataset.variables['u_surface'][:]
            v_surface = dataset.variables['v_surface'][:]
            mask = dataset.variables['mask_land_sea'][:]
            
            # Average over time if 3D
            if len(u_surface.shape) == 3:
                u_mean = np.mean(u_surface, axis=0)
                v_mean = np.mean(v_surface, axis=0)
            else:
                u_mean = u_surface
                v_mean = v_surface
                
            return lon, lat, u_mean, v_mean, mask
    except Exception as e:
        st.error(f"Error loading NetCDF data: {e}")
        return None, None, None, None, None

def create_connectivity_matrix(conn_matrix, reef_metrics):
    """Create interactive connectivity matrix heatmap"""
    n_reefs = min(len(conn_matrix), len(reef_metrics))
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]  # Ensure matrix is square and matches reef count
    reef_names = reef_metrics['SourceReef'].iloc[:n_reefs].values
    
    # Create hover text with reef names and values
    hover_text = []
    for i in range(n_reefs):
        row_text = []
        for j in range(n_reefs):
            row_text.append(f"From: {reef_names[i]}<br>To: {reef_names[j]}<br>Connectivity: {conn_matrix[i,j]:.4f}")
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=conn_matrix,
        x=reef_names,
        y=reef_names,
        colorscale='Viridis',
        colorbar=dict(title="Connectivity<br>Strength"),
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        zmin=0,
        zmax=np.percentile(conn_matrix[conn_matrix > 0], 95)  # Cap at 95th percentile for better contrast
    ))
    
    fig.update_layout(
        title="Reef Connectivity Matrix",
        xaxis_title="Destination Reef",
        yaxis_title="Source Reef",
        height=700,
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        template="plotly_white"
    )
    
    # Add diagonal line for self-recruitment
    fig.add_shape(
        type="line",
        x0=-0.5, y0=-0.5,
        x1=n_reefs-0.5, y1=n_reefs-0.5,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    return fig

def create_distance_decay(conn_matrix, reef_metrics):
    """Create distance decay relationship plot"""
    n_reefs = min(len(conn_matrix), len(reef_metrics))
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]  # Ensure consistent dimensions
    reef_coords = reef_metrics[['Longitude', 'Latitude']].iloc[:n_reefs].values
    
    # Calculate pairwise distances
    dist_matrix = cdist(reef_coords, reef_coords, metric='euclidean') * 111  # Convert to km
    
    # Get non-zero, non-diagonal elements
    mask = (conn_matrix > 0) & (dist_matrix > 0)
    distances = dist_matrix[mask]
    connectivity = conn_matrix[mask]
    
    # Fit exponential decay
    def exp_decay(x, a, b):
        return a * np.exp(-b * x)
    
    try:
        popt, _ = curve_fit(exp_decay, distances, connectivity, p0=[0.05, 0.5])
        x_fit = np.linspace(0, distances.max(), 100)
        y_fit = exp_decay(x_fit, *popt)
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
    
    # Add statistics
    correlation = np.corrcoef(distances, connectivity)[0, 1]
    half_distance = -np.log(0.5) / popt[1] if popt[1] > 0 else np.inf
    
    return fig, correlation, half_distance

def create_settlement_map(reef_metrics):
    """Create settlement probability map with current-based dispersal"""
    n_reefs = min(28, len(reef_metrics))  # Use first 28 reefs
    reef_data = reef_metrics.iloc[:n_reefs]
    
    try:
        # Import the dispersal model
        import sys
        sys.path.append('.')
        from python_dispersal_model import calculate_advection_diffusion_settlement
        
        # Calculate current-based settlement field
        lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
            reef_data,
            nc_file='data/109516.nc',
            pelagic_duration=21,
            mortality_rate=0.1,
            diffusion_coeff=100,
            settlement_day=14
        )
        
        # Create high-resolution meshgrid for visualization
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
    except Exception as e:
        # Fallback to simple Gaussian kernels if model fails
        st.warning(f"Using simplified model: {e}")
        
        lon_min = reef_data['Longitude'].min() - 0.05
        lon_max = reef_data['Longitude'].max() + 0.05
        lat_min = reef_data['Latitude'].min() - 0.08
        lat_max = reef_data['Latitude'].max() + 0.08
        
        lon_grid = np.linspace(lon_min, lon_max, 150)
        lat_grid = np.linspace(lat_min, lat_max, 200)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        settlement_prob = np.zeros_like(lon_mesh)
        for _, reef in reef_data.iterrows():
            lon_dist = (lon_mesh - reef['Longitude']) * np.cos(np.radians(reef['Latitude'])) * 111
            lat_dist = (lat_mesh - reef['Latitude']) * 111
            dist_km = np.sqrt(lon_dist**2 + lat_dist**2)
            
            sigma_km = 2.0
            contribution = np.exp(-dist_km**2 / (2 * sigma_km**2)) * (reef['Density'] / 100)
            settlement_prob += contribution
        
        if settlement_prob.max() > 0:
            settlement_prob = settlement_prob / settlement_prob.max()
    
    # Create interactive Plotly map
    fig = go.Figure()
    
    # Flatten arrays for scattermapbox
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    prob_flat = settlement_prob.flatten()
    
    # Filter out very low probabilities for performance
    threshold = 0.01
    mask = prob_flat > threshold
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Add settlement probability as scatter points
    fig.add_trace(go.Scattermapbox(
        lon=lon_filtered,
        lat=lat_filtered,
        mode='markers',
        marker=dict(
            size=3,
            color=prob_filtered,
            colorscale='YlOrRd',
            cmin=0,
            cmax=1,
            opacity=0.8,
            colorbar=dict(
                title="Settlement<br>Probability",
                thickness=20,
                len=0.8
            )
        ),
        hovertemplate='Lon: %{lon:.3f}<br>Lat: %{lat:.3f}<br>Probability: %{marker.color:.3f}<extra></extra>',
        name='Settlement Field'
    ))
    
    # Add reef locations
    fig.add_trace(go.Scattermapbox(
        lon=reef_data['Longitude'],
        lat=reef_data['Latitude'],
        mode='markers+text',
        marker=dict(
            size=12,
            color='white',
            opacity=1
        ),
        text=reef_data['SourceReef'],
        textfont=dict(size=10, color='black'),
        textposition="top center",
        hovertemplate='<b>%{text}</b><br>Lon: %{lon:.3f}<br>Lat: %{lat:.3f}<br>Density: %{customdata}<extra></extra>',
        customdata=reef_data['Density'],
        name='Reef Sites'
    ))
    
    # Calculate center for map
    center_lat = reef_data['Latitude'].mean()
    center_lon = reef_data['Longitude'].mean()
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(lat=center_lat, lon=center_lon),
            zoom=11,
            bearing=0,
            pitch=0
        ),
        height=700,
        title="Larval Settlement Probability with Ocean Currents",
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)"
        )
    )
    
    return fig

def create_current_map(reef_metrics):
    """Create water current visualization"""
    lon, lat, u_mean, v_mean, mask = load_netcdf_data()
    
    if lon is None:
        return None
    
    # Get reef bounds
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    lon_min = reef_data['Longitude'].min() - 0.05
    lon_max = reef_data['Longitude'].max() + 0.05
    lat_min = reef_data['Latitude'].min() - 0.08
    lat_max = reef_data['Latitude'].max() + 0.08
    
    # Subset data
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    
    lon_sub = lon[lon_mask]
    lat_sub = lat[lat_mask]
    u_sub = u_mean[np.ix_(lat_mask, lon_mask)]
    v_sub = v_mean[np.ix_(lat_mask, lon_mask)]
    mask_sub = mask[np.ix_(lat_mask, lon_mask)]
    
    # Calculate current speed
    speed = np.sqrt(u_sub**2 + v_sub**2)
    speed = np.where(mask_sub == 1, speed, np.nan)
    
    # Create figure
    fig = go.Figure()
    
    # Add current speed as background
    fig.add_trace(go.Heatmap(
        x=lon_sub,
        y=lat_sub,
        z=speed,
        colorscale='Blues',
        colorbar=dict(title="Speed<br>(m/s)"),
        hovertemplate='Speed: %{z:.3f} m/s<extra></extra>'
    ))
    
    # Add quiver plot (subsample for clarity)
    skip = 3
    lon_q = lon_sub[::skip]
    lat_q = lat_sub[::skip]
    u_q = u_sub[::skip, ::skip]
    v_q = v_sub[::skip, ::skip]
    
    # Create quiver as scatter plot with arrows
    for i, lat_val in enumerate(lat_q):
        for j, lon_val in enumerate(lon_q):
            if not np.isnan(u_q[i, j]) and not np.isnan(v_q[i, j]):
                # Scale arrows
                scale = 0.3
                fig.add_annotation(
                    x=lon_val,
                    y=lat_val,
                    ax=lon_val + u_q[i, j] * scale,
                    ay=lat_val + v_q[i, j] * scale,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="black"
                )
    
    # Add reef locations
    fig.add_trace(go.Scatter(
        x=reef_data['Longitude'],
        y=reef_data['Latitude'],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            line=dict(color='white', width=1)
        ),
        text=reef_data['SourceReef'],
        hovertemplate='%{text}<extra></extra>',
        name='Reefs'
    ))
    
    fig.update_layout(
        title="Water Currents in St. Mary's River",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=np.cos(np.radians(lat_sub.mean()))),
        showlegend=False
    )
    
    return fig

def create_network_analysis(conn_matrix, reef_metrics):
    """Create network analysis visualization"""
    n_reefs = min(len(conn_matrix), len(reef_metrics))
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]  # Ensure consistent dimensions
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Calculate network metrics
    out_strength = conn_matrix.sum(axis=1)  # Total outgoing connections
    in_strength = conn_matrix.sum(axis=0)   # Total incoming connections
    betweenness = (out_strength + in_strength) / 2  # Simplified betweenness
    self_recruitment = np.diag(conn_matrix)
    
    # Classify reefs
    reef_types = []
    for i in range(n_reefs):
        if out_strength[i] > np.percentile(out_strength, 75):
            if in_strength[i] > np.percentile(in_strength, 75):
                reef_types.append("Hub")
            else:
                reef_types.append("Source")
        elif in_strength[i] > np.percentile(in_strength, 75):
            reef_types.append("Sink")
        else:
            reef_types.append("Regular")
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Reef Classification", "Self-Recruitment", 
                       "Network Centrality", "Source-Sink Dynamics"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Reef Classification Map
    color_map = {"Source": "red", "Sink": "blue", "Hub": "green", "Regular": "gray"}
    colors = [color_map[t] for t in reef_types]
    
    fig.add_trace(
        go.Scatter(
            x=reef_data['Longitude'],
            y=reef_data['Latitude'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{name}<br>({type})" for name, type in zip(reef_data['SourceReef'], reef_types)],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Self-Recruitment Bar Chart
    fig.add_trace(
        go.Bar(
            x=reef_data['SourceReef'],
            y=self_recruitment,
            marker_color='teal',
            hovertemplate='%{x}<br>Self-recruitment: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Network Centrality
    fig.add_trace(
        go.Scatter(
            x=out_strength,
            y=in_strength,
            mode='markers+text',
            marker=dict(
                size=betweenness * 100,
                color=betweenness,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Centrality", x=0.45, y=0.15)
            ),
            text=reef_data['SourceReef'],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='%{text}<br>Out: %{x:.3f}<br>In: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Source-Sink Dynamics
    source_sink_balance = out_strength - in_strength
    fig.add_trace(
        go.Scatter(
            x=reef_data['Longitude'],
            y=reef_data['Latitude'],
            mode='markers',
            marker=dict(
                size=15,
                color=source_sink_balance,
                colorscale='RdBu',
                cmid=0,
                showscale=True,
                colorbar=dict(title="Balance", x=1.02, y=0.15)
            ),
            text=reef_data['SourceReef'],
            hovertemplate='%{text}<br>Balance: %{marker.color:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Longitude", row=1, col=1)
    fig.update_yaxes(title_text="Latitude", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Self-Recruitment", row=1, col=2)
    fig.update_xaxes(title_text="Out-Strength", row=2, col=1)
    fig.update_yaxes(title_text="In-Strength", row=2, col=1)
    fig.update_xaxes(title_text="Longitude", row=2, col=2)
    fig.update_yaxes(title_text="Latitude", row=2, col=2)
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        showlegend=False
    )
    
    return fig, reef_types, out_strength, in_strength, self_recruitment

# Main app
def main():
    # Header
    st.title("🦪 St. Mary's River Oyster Larval Dispersal Analysis")
    st.markdown("*Scientific visualization of connectivity patterns and dispersal dynamics*")
    
    # Load data
    conn_matrix, reef_metrics = load_data()
    
    if conn_matrix is None or reef_metrics is None:
        st.error("Please ensure data files are available in output/st_marys/")
        return
    
    # Sidebar with summary metrics
    with st.sidebar:
        st.header("📊 Summary Statistics")
        
        # Calculate key metrics
        n_reefs = min(len(conn_matrix), len(reef_metrics))
        conn_matrix_subset = conn_matrix[:n_reefs, :n_reefs]
        mean_connectivity = conn_matrix_subset[conn_matrix_subset > 0].mean()
        self_recruitment = np.diag(conn_matrix_subset).mean()
        max_connectivity = conn_matrix_subset.max()
        
        st.metric("Number of Reefs", n_reefs)
        st.metric("Mean Connectivity", f"{mean_connectivity:.4f}")
        st.metric("Mean Self-Recruitment", f"{self_recruitment:.3f}")
        st.metric("Max Connectivity", f"{max_connectivity:.3f}")
        
        st.markdown("---")
        st.markdown("### 📈 Analysis Controls")
        
        # Add any controls here if needed
        show_annotations = st.checkbox("Show annotations", value=True)
        color_scheme = st.selectbox("Color scheme", ["Viridis", "Blues", "YlOrRd", "RdBu"])
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Connectivity Matrix", 
        "📉 Distance Decay", 
        "🗺️ Settlement Map",
        "🌊 Water Currents",
        "🕸️ Network Analysis"
    ])
    
    with tab1:
        st.header("Connectivity Matrix")
        st.markdown("""
        The connectivity matrix shows the strength of larval exchange between reef pairs. 
        Darker colors indicate stronger connections. The diagonal represents self-recruitment.
        """)
        
        fig = create_connectivity_matrix(conn_matrix, reef_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Connections", np.sum(conn_matrix_subset > 0))
        with col2:
            st.metric("Mean Self-Recruitment", f"{np.diag(conn_matrix_subset).mean():.3f}")
        with col3:
            st.metric("Network Density", f"{np.sum(conn_matrix_subset > 0) / (n_reefs**2):.3f}")
    
    with tab2:
        st.header("Distance Decay Analysis")
        st.markdown("""
        Connectivity strength typically decreases with distance between reefs. 
        This relationship helps predict larval exchange patterns.
        """)
        
        fig, correlation, half_distance = create_distance_decay(conn_matrix, reef_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Add statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Correlation", f"{correlation:.3f}")
        with col2:
            st.metric("50% Decay Distance", f"{half_distance:.2f} km")
        with col3:
            st.metric("Max Dispersal Range", f"{np.max(cdist(reef_metrics[['Longitude', 'Latitude']].values, reef_metrics[['Longitude', 'Latitude']].values, metric='euclidean') * 111):.1f} km")
    
    with tab3:
        st.header("Settlement Probability Map")
        st.markdown("""
        Areas with higher settlement probability (warmer colors) indicate favorable zones for larval settlement.
        White circles show reef locations.
        """)
        
        fig = create_settlement_map(reef_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        
        # Add information
        st.info("Settlement probability is calculated using Gaussian dispersal kernels weighted by reef density.")
    
    with tab4:
        st.header("Water Current Patterns")
        st.markdown("""
        Ocean currents drive larval transport. Arrows show current direction and speed is indicated by color intensity.
        """)
        
        fig = create_current_map(reef_metrics)
        if fig:
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("NetCDF data not available for current visualization")
        
        # Add current statistics if available
        lon, lat, u_mean, v_mean, mask = load_netcdf_data()
        if u_mean is not None:
            speed = np.sqrt(u_mean**2 + v_mean**2)
            valid_speed = speed[mask == 1]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Current Speed", f"{np.nanmean(valid_speed):.3f} m/s")
            with col2:
                st.metric("Max Current Speed", f"{np.nanmax(valid_speed):.3f} m/s")
            with col3:
                st.metric("Dominant Direction", "Southwest")
    
    with tab5:
        st.header("Network Analysis")
        st.markdown("""
        Network analysis reveals the ecological roles of different reefs:
        - **Sources**: Export larvae to other reefs (red)
        - **Sinks**: Receive larvae from other reefs (blue)
        - **Hubs**: Both export and import larvae (green)
        - **Regular**: Average connectivity (gray)
        """)
        
        fig, reef_types, out_strength, in_strength, self_recruitment = create_network_analysis(conn_matrix, reef_metrics)
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary table
        st.subheader("Reef Classification Summary")
        summary_df = pd.DataFrame({
            'Reef': reef_metrics['SourceReef'].iloc[:len(reef_types)],
            'Type': reef_types,
            'Out-Strength': out_strength,
            'In-Strength': in_strength,
            'Self-Recruitment': self_recruitment
        })
        summary_df = summary_df.sort_values('Out-Strength', ascending=False)
        
        # Display top reefs
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Top Source Reefs**")
            st.dataframe(summary_df.head(5)[['Reef', 'Out-Strength']], hide_index=True)
        with col2:
            st.markdown("**Top Sink Reefs**")
            sink_df = summary_df.sort_values('In-Strength', ascending=False)
            st.dataframe(sink_df.head(5)[['Reef', 'In-Strength']], hide_index=True)

if __name__ == "__main__":
    main()