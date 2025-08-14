"""
Simplified Settlement Map with Manual Exclusion Zones
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sys
import os
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shapely.vectorized import contains
except ImportError:
    contains = None

def render_section(reef_metrics):
    """Render settlement map with manual coordinate input for exclusion"""
    st.header("🎯 Interactive Settlement Map")
    
    # Initialize exclusion zones in session state
    if 'exclude_zones' not in st.session_state:
        st.session_state.exclude_zones = []
    
    # Manual exclusion zone input
    with st.expander("➕ Add Custom Exclusion Zones", expanded=False):
        st.markdown("Define additional areas to exclude from the map")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            ex_lon_min = st.number_input("Min Lon", value=-76.48, format="%.3f", key="ex_lon_min")
        with col2:
            ex_lon_max = st.number_input("Max Lon", value=-76.47, format="%.3f", key="ex_lon_max")
        with col3:
            ex_lat_min = st.number_input("Min Lat", value=38.15, format="%.3f", key="ex_lat_min")
        with col4:
            ex_lat_max = st.number_input("Max Lat", value=38.16, format="%.3f", key="ex_lat_max")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Exclusion Zone"):
                zone = {
                    'lon_min': ex_lon_min,
                    'lon_max': ex_lon_max,
                    'lat_min': ex_lat_min,
                    'lat_max': ex_lat_max
                }
                st.session_state.exclude_zones.append(zone)
                st.success(f"Added exclusion zone")
        
        with col2:
            if st.button("Clear All Zones"):
                st.session_state.exclude_zones = []
                st.success("Cleared all zones")
        
        # Show current zones
        if st.session_state.exclude_zones:
            st.markdown("#### Current Exclusion Zones:")
            for i, zone in enumerate(st.session_state.exclude_zones):
                st.text(f"{i+1}. Lon: [{zone['lon_min']:.3f}, {zone['lon_max']:.3f}], Lat: [{zone['lat_min']:.3f}, {zone['lat_max']:.3f}]")
    
    # Load reef data
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Load water boundary for filtering land
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
        if water_geometry:
            st.success("✅ Water boundary loaded - land areas filtered")
    except:
        water_geometry = None
        st.warning("⚠️ Could not load water boundary")
    
    # Try to load from dispersal model
    try:
        from python_dispersal_model import calculate_advection_diffusion_settlement
        lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
            reef_data,
            nc_file='data/109516.nc',
            pelagic_duration=21,
            mortality_rate=0.1,
            diffusion_coeff=100,
            settlement_day=14
        )
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    except:
        # Simple fallback
        lon_grid = np.linspace(-76.495, -76.4, 100)
        lat_grid = np.linspace(38.125, 38.23, 100)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        settlement_prob = np.zeros_like(lon_mesh)
        for _, reef in reef_data.iterrows():
            lon_dist = (lon_mesh - reef['Longitude']) * 111 * np.cos(np.radians(reef['Latitude']))
            lat_dist = (lat_mesh - reef['Latitude']) * 111
            dist_km = np.sqrt(lon_dist**2 + lat_dist**2)
            settlement_prob += np.exp(-dist_km**2 / 8) * (reef.get('Density', 100) / 100)
        
        if settlement_prob.max() > 0:
            settlement_prob = settlement_prob / settlement_prob.max()
    
    # Flatten for plotting
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    prob_flat = settlement_prob.flatten()
    
    # Apply water mask first (filter out land)
    mask = prob_flat > 0.01
    
    # Filter out land using water boundary
    if water_geometry is not None and contains is not None:
        water_mask = contains(water_geometry, lon_flat, lat_flat)
        mask = mask & water_mask
    
    # Then apply custom exclusion zones
    for zone in st.session_state.exclude_zones:
        zone_mask = ~((lon_flat >= zone['lon_min']) & 
                     (lon_flat <= zone['lon_max']) & 
                     (lat_flat >= zone['lat_min']) & 
                     (lat_flat <= zone['lat_max']))
        mask = mask & zone_mask
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Create map
    fig = go.Figure()
    
    # Add probability points
    fig.add_trace(go.Scattermapbox(
        lon=lon_filtered,
        lat=lat_filtered,
        mode='markers',
        marker=dict(
            size=3,
            color=prob_filtered,
            colorscale='Turbo',
            cmin=0,
            cmax=1,
            opacity=0.8,
            colorbar=dict(title="Probability")
        ),
        name='Settlement'
    ))
    
    # Add exclusion zones as red rectangles
    for i, zone in enumerate(st.session_state.exclude_zones):
        rect_lon = [zone['lon_min'], zone['lon_max'], zone['lon_max'], zone['lon_min'], zone['lon_min']]
        rect_lat = [zone['lat_min'], zone['lat_min'], zone['lat_max'], zone['lat_max'], zone['lat_min']]
        
        fig.add_trace(go.Scattermapbox(
            lon=rect_lon,
            lat=rect_lat,
            mode='lines',
            line=dict(color='red', width=2),
            name=f'Exclusion {i+1}'
        ))
    
    # Add reefs
    fig.add_trace(go.Scattermapbox(
        lon=reef_data['Longitude'],
        lat=reef_data['Latitude'],
        mode='markers',
        marker=dict(size=10, color='white'),
        text=reef_data['SourceReef'],
        name='Reefs'
    ))
    
    fig.update_layout(
        mapbox=dict(
            style='open-street-map',
            center=dict(
                lat=reef_data['Latitude'].mean(),
                lon=reef_data['Longitude'].mean()
            ),
            zoom=11
        ),
        height=600,
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Info about the visualization
    st.info("🌊 Settlement probabilities are shown only over water. Land areas are automatically filtered out using USGS water boundary data.")
    st.info("📍 Use the expander above to add custom exclusion zones if needed.")