"""
Settlement Map with Saved Exclusion Zones Management
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json
import os
import sys
# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from shapely.vectorized import contains
except ImportError:
    contains = None

# File to store exclusion zones
EXCLUSION_ZONES_FILE = 'data/exclusion_zones.json'

def load_exclusion_zones():
    """Load saved exclusion zones from file"""
    if os.path.exists(EXCLUSION_ZONES_FILE):
        try:
            with open(EXCLUSION_ZONES_FILE, 'r') as f:
                zones = json.load(f)
                return zones
        except:
            return []
    return []

def save_exclusion_zones(zones):
    """Save exclusion zones to file"""
    os.makedirs(os.path.dirname(EXCLUSION_ZONES_FILE), exist_ok=True)
    with open(EXCLUSION_ZONES_FILE, 'w') as f:
        json.dump(zones, f, indent=2)

def render_section(reef_metrics):
    """Render settlement map with exclusion zone management"""
    st.header("🎯 Interactive Settlement Map")
    
    # Initialize session state with saved zones
    if 'exclude_zones' not in st.session_state:
        st.session_state.exclude_zones = load_exclusion_zones()
    
    # Zone Management Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Zones"):
            save_exclusion_zones(st.session_state.exclude_zones)
            st.success(f"Saved {len(st.session_state.exclude_zones)} zones")
    
    with col2:
        if st.button("🔄 Reload Zones"):
            st.session_state.exclude_zones = load_exclusion_zones()
            st.success("Reloaded zones from file")
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.exclude_zones = []
            save_exclusion_zones([])
            st.success("Cleared all zones")
            st.rerun()
    
    # Add new exclusion zone
    with st.expander("➕ Add Exclusion Zone", expanded=False):
        st.markdown("Define a rectangular area to exclude from the settlement map")
        
        col1, col2 = st.columns(2)
        with col1:
            lon_min = st.number_input("Min Longitude", value=-76.48, format="%.4f", step=0.001, key="add_lon_min")
            lat_min = st.number_input("Min Latitude", value=38.15, format="%.4f", step=0.001, key="add_lat_min")
        
        with col2:
            lon_max = st.number_input("Max Longitude", value=-76.47, format="%.4f", step=0.001, key="add_lon_max")
            lat_max = st.number_input("Max Latitude", value=38.16, format="%.4f", step=0.001, key="add_lat_max")
        
        if st.button("Add Zone", type="primary"):
            zone = {
                'lon_min': float(lon_min),
                'lon_max': float(lon_max),
                'lat_min': float(lat_min),
                'lat_max': float(lat_max)
            }
            st.session_state.exclude_zones.append(zone)
            save_exclusion_zones(st.session_state.exclude_zones)
            st.success("Added exclusion zone!")
            st.rerun()
    
    # Show current zones
    if st.session_state.exclude_zones:
        st.markdown("### 📍 Current Exclusion Zones")
        for i, zone in enumerate(st.session_state.exclude_zones):
            col1, col2 = st.columns([5, 1])
            with col1:
                st.text(f"{i+1}. Lon: [{zone['lon_min']:.4f}, {zone['lon_max']:.4f}], "
                       f"Lat: [{zone['lat_min']:.4f}, {zone['lat_max']:.4f}]")
            with col2:
                if st.button("❌", key=f"remove_{i}", help=f"Remove zone {i+1}"):
                    st.session_state.exclude_zones.pop(i)
                    save_exclusion_zones(st.session_state.exclude_zones)
                    st.rerun()
    
    # Load reef data
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Load water boundary
    water_geometry = None
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
        if water_geometry and 'water_loaded' not in st.session_state:
            st.success("✅ Water boundary loaded - land filtered")
            st.session_state.water_loaded = True
    except:
        pass
    
    # Calculate settlement field
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
    except Exception as e:
        # Fallback
        st.warning(f"Using simplified model: {str(e)[:100]}")
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
    
    # Flatten arrays
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    prob_flat = settlement_prob.flatten()
    
    # Apply filters
    mask = prob_flat > 0.01
    
    # Water mask
    if water_geometry is not None and contains is not None:
        water_mask = contains(water_geometry, lon_flat, lat_flat)
        mask = mask & water_mask
    
    # Exclusion zones
    for zone in st.session_state.exclude_zones:
        zone_mask = ~((lon_flat >= zone['lon_min']) & 
                     (lon_flat <= zone['lon_max']) & 
                     (lat_flat >= zone['lat_min']) & 
                     (lat_flat <= zone['lat_max']))
        mask = mask & zone_mask
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Transform for better visualization
    prob_transformed = np.sqrt(prob_filtered)
    
    # Create map
    fig = go.Figure()
    
    # Settlement probability
    if len(prob_transformed) > 0:
        p25 = np.percentile(prob_transformed, 25)
        p75 = np.percentile(prob_transformed, 75)
        iqr = p75 - p25
        color_min = max(0, p25 - 0.5 * iqr)
        color_max = min(1, p75 + 0.5 * iqr)
    else:
        color_min, color_max = 0, 1
    
    fig.add_trace(go.Scattermapbox(
        lon=lon_filtered,
        lat=lat_filtered,
        mode='markers',
        marker=dict(
            size=3,
            color=prob_transformed,
            colorscale='Turbo',
            cmin=color_min,
            cmax=color_max,
            opacity=0.9,
            colorbar=dict(title="Settlement<br>Probability", thickness=20)
        ),
        name='Settlement',
        hovertemplate='Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<br>Prob: %{customdata:.3f}<extra></extra>',
        customdata=prob_filtered
    ))
    
    # Add exclusion zones
    for i, zone in enumerate(st.session_state.exclude_zones):
        rect_lon = [zone['lon_min'], zone['lon_max'], zone['lon_max'], zone['lon_min'], zone['lon_min']]
        rect_lat = [zone['lat_min'], zone['lat_min'], zone['lat_max'], zone['lat_max'], zone['lat_min']]
        
        fig.add_trace(go.Scattermapbox(
            lon=rect_lon,
            lat=rect_lat,
            mode='lines',
            fill='toself',
            fillcolor='rgba(255, 0, 0, 0.2)',
            line=dict(color='red', width=2),
            name=f'Zone {i+1}',
            hoverinfo='name'
        ))
    
    # Add reefs
    fig.add_trace(go.Scattermapbox(
        lon=reef_data['Longitude'],
        lat=reef_data['Latitude'],
        mode='markers+text',
        marker=dict(size=10, color='white'),
        text=reef_data['SourceReef'],
        textfont=dict(size=8),
        textposition="top center",
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
    
    # Instructions
    with st.expander("📖 How to Define Exclusion Zones"):
        st.markdown("""
        1. Click **"➕ Add Exclusion Zone"** above
        2. Enter the min/max longitude and latitude values
        3. Click **"Add Zone"** to create the exclusion
        4. Red rectangles show excluded areas on the map
        5. Click **"💾 Save Zones"** to persist your changes
        
        **Tips:**
        - Zoom into the map to identify coordinates
        - Hover over the map to see lon/lat values
        - Exclusion zones are shared with the regular Settlement Map
        - Zones are saved to `data/exclusion_zones.json`
        """)
    
    st.info(f"📁 Exclusion zones file: {EXCLUSION_ZONES_FILE}")
    st.info("🌊 Land areas are automatically filtered using water boundary data")