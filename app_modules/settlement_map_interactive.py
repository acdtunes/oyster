"""
Interactive Settlement Map with Rectangle Selection for Exclusion Zones
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sys
import os
sys.path.append('.')
from shapely.vectorized import contains
from shapely.geometry import Polygon, box

def create_interactive_visualization(reef_metrics):
    """Create settlement probability map with interactive rectangle selection"""
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Initialize session state for exclusion zones
    if 'exclusion_zones' not in st.session_state:
        st.session_state.exclusion_zones = []
    if 'adding_zone' not in st.session_state:
        st.session_state.adding_zone = False
    
    # Add controls for rectangle selection
    st.markdown("### 🎯 Define Exclusion Zones")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("✏️ Add Exclusion Zone"):
            st.session_state.adding_zone = True
            
    with col2:
        if st.button("🗑️ Clear All Zones"):
            st.session_state.exclusion_zones = []
            st.session_state.adding_zone = False
            st.rerun()
            
    with col3:
        if st.button("💾 Save Zones"):
            # Save exclusion zones to file
            import json
            with open('data/exclusion_zones.json', 'w') as f:
                json.dump(st.session_state.exclusion_zones, f)
            st.success("Zones saved!")
    
    # Manual input for rectangle coordinates
    if st.session_state.get('adding_zone', False):
        st.markdown("#### Enter Rectangle Coordinates")
        col1, col2 = st.columns(2)
        
        with col1:
            lon_min = st.number_input("Min Longitude", value=-76.48, format="%.4f", key="rect_lon_min")
            lat_min = st.number_input("Min Latitude", value=38.15, format="%.4f", key="rect_lat_min")
            
        with col2:
            lon_max = st.number_input("Max Longitude", value=-76.47, format="%.4f", key="rect_lon_max")
            lat_max = st.number_input("Max Latitude", value=38.16, format="%.4f", key="rect_lat_max")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Add Rectangle"):
                rect = {
                    'lon_min': lon_min,
                    'lon_max': lon_max,
                    'lat_min': lat_min,
                    'lat_max': lat_max
                }
                st.session_state.exclusion_zones.append(rect)
                st.session_state.adding_zone = False
                st.rerun()
                
        with col2:
            if st.button("❌ Cancel"):
                st.session_state.adding_zone = False
                st.rerun()
    
    # Display current exclusion zones
    if st.session_state.exclusion_zones:
        st.markdown("#### Current Exclusion Zones")
        for i, zone in enumerate(st.session_state.exclusion_zones):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.text(f"Zone {i+1}: [{zone['lon_min']:.3f}, {zone['lat_min']:.3f}] to [{zone['lon_max']:.3f}, {zone['lat_max']:.3f}]")
            with col2:
                if st.button(f"Remove", key=f"remove_{i}"):
                    st.session_state.exclusion_zones.pop(i)
                    st.rerun()
    
    try:
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
        
        # Create meshgrid for visualization
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
    except Exception as e:
        # Fallback to simple Gaussian kernels if model fails
        st.warning(f"Using simplified model: {e}")
        
        lon_min = -76.495
        lon_max = -76.4
        lat_min = 38.125
        lat_max = 38.23
        
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
    
    # Load water boundary
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
    except:
        water_geometry = None
    
    # Flatten arrays for scattermapbox
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    prob_flat = settlement_prob.flatten()
    
    # Apply exclusion zones
    mask = prob_flat > 0.01  # Initial threshold
    
    # Filter out exclusion zones
    for zone in st.session_state.exclusion_zones:
        zone_mask = ~((lon_flat >= zone['lon_min']) & (lon_flat <= zone['lon_max']) & 
                     (lat_flat >= zone['lat_min']) & (lat_flat <= zone['lat_max']))
        mask = mask & zone_mask
    
    # Apply water mask if available
    if water_geometry is not None:
        from shapely.vectorized import contains
        water_mask_flat = contains(water_geometry, lon_flat, lat_flat)
        mask = mask & water_mask_flat
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Use power transformation to enhance contrast
    prob_transformed = np.sqrt(prob_filtered)
    
    # Calculate percentiles for better color scaling
    if len(prob_transformed) > 0:
        p25 = np.percentile(prob_transformed, 25)
        p75 = np.percentile(prob_transformed, 75)
        iqr = p75 - p25
        color_min = max(0, p25 - 0.5 * iqr)
        color_max = min(1, p75 + 0.5 * iqr)
        color_values = prob_transformed
    else:
        color_min = 0
        color_max = 1
        color_values = prob_filtered
    
    # Add settlement probability as scatter points
    fig.add_trace(go.Scattermapbox(
        lon=lon_filtered,
        lat=lat_filtered,
        mode='markers',
        marker=dict(
            size=3,
            color=color_values,
            colorscale='Turbo',
            cmin=color_min,
            cmax=color_max,
            opacity=0.9,
            colorbar=dict(
                title="Settlement<br>Probability",
                thickness=20,
                len=0.8
            )
        ),
        hovertemplate='Lon: %{lon:.3f}<br>Lat: %{lat:.3f}<br>Probability: %{customdata:.3f}<extra></extra>',
        customdata=prob_filtered,
        name='Settlement Field'
    ))
    
    # Add exclusion zone rectangles
    for i, zone in enumerate(st.session_state.exclusion_zones):
        # Draw rectangle outline
        rect_lon = [zone['lon_min'], zone['lon_max'], zone['lon_max'], zone['lon_min'], zone['lon_min']]
        rect_lat = [zone['lat_min'], zone['lat_min'], zone['lat_max'], zone['lat_max'], zone['lat_min']]
        
        fig.add_trace(go.Scattermapbox(
            lon=rect_lon,
            lat=rect_lat,
            mode='lines',
            line=dict(color='red', width=3),
            name=f'Exclusion Zone {i+1}',
            hoverinfo='name'
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
        title="Interactive Settlement Map with Exclusion Zones",
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

def render_section(reef_metrics):
    """Render the complete interactive settlement map section"""
    st.header("🗺️ Interactive Settlement Probability Map")
    st.markdown("""
    Define exclusion zones to remove areas from the settlement probability calculation.
    Use the controls above to add rectangles that will be filtered out from the map.
    """)
    
    # Add instructions
    with st.expander("📖 How to use"):
        st.markdown("""
        1. Click **Add Exclusion Zone** to define a new rectangle
        2. Enter the min/max longitude and latitude for the rectangle
        3. Click **Add Rectangle** to apply the exclusion zone
        4. The red rectangles show excluded areas
        5. Use **Clear All Zones** to reset
        6. Use **Save Zones** to persist your selections
        """)
    
    fig = create_interactive_visualization(reef_metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    # Option to load/save coordinates from clipboard
    st.markdown("---")
    st.markdown("### 📋 Bulk Import Zones")
    
    zones_text = st.text_area(
        "Paste rectangle coordinates (format: lon_min,lat_min,lon_max,lat_max per line)",
        height=100,
        placeholder="-76.480,38.150,-76.470,38.160\n-76.460,38.140,-76.450,38.150"
    )
    
    if st.button("Import Zones from Text"):
        try:
            lines = zones_text.strip().split('\n')
            new_zones = []
            for line in lines:
                if line.strip():
                    coords = line.split(',')
                    if len(coords) == 4:
                        zone = {
                            'lon_min': float(coords[0]),
                            'lat_min': float(coords[1]),
                            'lon_max': float(coords[2]),
                            'lat_max': float(coords[3])
                        }
                        new_zones.append(zone)
            
            if new_zones:
                st.session_state.exclusion_zones.extend(new_zones)
                st.success(f"Added {len(new_zones)} exclusion zones!")
                st.rerun()
        except Exception as e:
            st.error(f"Error parsing zones: {e}")