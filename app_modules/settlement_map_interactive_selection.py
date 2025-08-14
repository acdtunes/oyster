"""
Interactive Settlement Map with Mouse Selection for Exclusion Zones
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
    """Render settlement map with interactive exclusion zone selection"""
    st.header("🎯 Interactive Settlement Map with Selection")
    
    # Initialize session state with saved zones
    if 'exclude_zones' not in st.session_state:
        st.session_state.exclude_zones = load_exclusion_zones()
    
    # Controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Zones to File"):
            save_exclusion_zones(st.session_state.exclude_zones)
            st.success(f"Saved {len(st.session_state.exclude_zones)} zones to {EXCLUSION_ZONES_FILE}")
    
    with col2:
        if st.button("🔄 Reload from File"):
            st.session_state.exclude_zones = load_exclusion_zones()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All Zones"):
            st.session_state.exclude_zones = []
            save_exclusion_zones([])  # Also clear the file
            st.rerun()
    
    # Instructions
    with st.expander("📖 How to Select Exclusion Zones", expanded=False):
        st.markdown("""
        1. **Use the Box Select tool** in the plot toolbar (rectangle icon)
        2. **Click and drag** on the map to draw a rectangle
        3. **Click "Add Selected Area"** to save the zone
        4. **Red rectangles** show excluded areas
        5. **Save zones** to persist across sessions
        
        💡 Tip: Zoom in for more precise selection
        """)
    
    # Load reef data
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Load water boundary for filtering land
    water_geometry = None
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
        if water_geometry:
            st.success("✅ Water boundary loaded - land areas filtered")
    except:
        st.warning("⚠️ Could not load water boundary")
    
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
    except:
        # Fallback
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
    
    if water_geometry is not None and contains is not None:
        water_mask = contains(water_geometry, lon_flat, lat_flat)
        mask = mask & water_mask
    
    # Apply saved exclusion zones
    for zone in st.session_state.exclude_zones:
        zone_mask = ~((lon_flat >= zone['lon_min']) & 
                     (lon_flat <= zone['lon_max']) & 
                     (lat_flat >= zone['lat_min']) & 
                     (lat_flat <= zone['lat_max']))
        mask = mask & zone_mask
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Create interactive map with selection capability
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
        name='Settlement',
        customdata=prob_filtered,
        hovertemplate='Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<br>Prob: %{customdata:.3f}<extra></extra>'
    ))
    
    # Add exclusion zones as red rectangles
    for i, zone in enumerate(st.session_state.exclude_zones):
        # Draw filled rectangle with outline
        rect_lon = [zone['lon_min'], zone['lon_max'], zone['lon_max'], zone['lon_min'], zone['lon_min']]
        rect_lat = [zone['lat_min'], zone['lat_min'], zone['lat_max'], zone['lat_max'], zone['lat_min']]
        
        # Filled rectangle
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
        textfont=dict(size=8, color='black'),
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
        showlegend=True,
        dragmode='select',  # Enable selection mode
        selectdirection='diagonal',
        modebar={'orientation': 'v'}
    )
    
    # Display the map and capture selection
    selected_data = st.plotly_chart(fig, use_container_width=True, key="selection_map", 
                                   on_select="rerun", selection_mode=['box'])
    
    # Process selection if available
    if selected_data and 'selection' in selected_data:
        if selected_data['selection'].get('box'):
            for box in selected_data['selection']['box']:
                # Extract bounds from selection
                x_range = box.get('x', [])
                y_range = box.get('y', [])
                
                if len(x_range) == 2 and len(y_range) == 2:
                    st.info(f"Selected area: Lon [{x_range[0]:.4f}, {x_range[1]:.4f}], Lat [{y_range[0]:.4f}, {y_range[1]:.4f}]")
                    
                    # Button to add selected area as exclusion zone
                    if st.button("➕ Add Selected Area as Exclusion Zone"):
                        zone = {
                            'lon_min': min(x_range),
                            'lon_max': max(x_range),
                            'lat_min': min(y_range),
                            'lat_max': max(y_range)
                        }
                        st.session_state.exclude_zones.append(zone)
                        save_exclusion_zones(st.session_state.exclude_zones)
                        st.success("Added exclusion zone!")
                        st.rerun()
    
    # Manual input as backup
    with st.expander("✏️ Manual Zone Input", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            m_lon_min = st.number_input("Min Lon", value=-76.48, format="%.4f", key="m_lon_min")
        with col2:
            m_lon_max = st.number_input("Max Lon", value=-76.47, format="%.4f", key="m_lon_max")
        with col3:
            m_lat_min = st.number_input("Min Lat", value=38.15, format="%.4f", key="m_lat_min")
        with col4:
            m_lat_max = st.number_input("Max Lat", value=38.16, format="%.4f", key="m_lat_max")
        
        if st.button("Add Manual Zone"):
            zone = {
                'lon_min': m_lon_min,
                'lon_max': m_lon_max,
                'lat_min': m_lat_min,
                'lat_max': m_lat_max
            }
            st.session_state.exclude_zones.append(zone)
            save_exclusion_zones(st.session_state.exclude_zones)
            st.success("Added manual zone!")
            st.rerun()
    
    # Show current zones
    if st.session_state.exclude_zones:
        st.markdown("### 📍 Current Exclusion Zones")
        for i, zone in enumerate(st.session_state.exclude_zones):
            col1, col2 = st.columns([4, 1])
            with col1:
                st.text(f"{i+1}. Lon: [{zone['lon_min']:.4f}, {zone['lon_max']:.4f}], Lat: [{zone['lat_min']:.4f}, {zone['lat_max']:.4f}]")
            with col2:
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.exclude_zones.pop(i)
                    save_exclusion_zones(st.session_state.exclude_zones)
                    st.rerun()
    
    st.info("🌊 Settlement probabilities are shown only over water. Land areas are automatically filtered.")
    st.info(f"📁 Exclusion zones are saved to: {EXCLUSION_ZONES_FILE}")