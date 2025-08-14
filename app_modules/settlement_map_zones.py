"""
Settlement Map with Efficient Polygon Exclusion Zones
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File to store polygon exclusion zones
EXCLUSION_ZONES_FILE = 'data/exclusion_polygons.json'

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

def points_in_polygon(points_x, points_y, poly_coords):
    """
    Efficient vectorized point-in-polygon test using ray casting algorithm
    """
    poly_x = np.array([c[0] for c in poly_coords])
    poly_y = np.array([c[1] for c in poly_coords])
    n = len(poly_coords) - 1  # Assuming closed polygon
    
    inside = np.zeros(len(points_x), dtype=bool)
    
    # Ray casting algorithm - vectorized
    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        # Check if point is on the right side of edge
        intersect = ((yi > points_y) != (yj > points_y)) & \
                   (points_x < (xj - xi) * (points_y - yi) / (yj - yi) + xi)
        inside = np.logical_xor(inside, intersect)
    
    return inside

def render_section(reef_metrics):
    """Render settlement map with efficient exclusion zones"""
    st.header("🎯 Settlement Map with Exclusion Zones")
    
    # Initialize session state
    if 'exclusion_polygons' not in st.session_state:
        st.session_state.exclusion_polygons = load_exclusion_zones()
    
    # Load reef data
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Control buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Save Zones"):
            save_exclusion_zones(st.session_state.exclusion_polygons)
            st.success(f"Saved {len(st.session_state.exclusion_polygons)} zones")
    
    with col2:
        if st.button("🔄 Reload"):
            st.session_state.exclusion_polygons = load_exclusion_zones()
            st.rerun()
    
    with col3:
        if st.button("🗑️ Clear All"):
            st.session_state.exclusion_polygons = []
            save_exclusion_zones([])
            st.rerun()
    
    # Simple rectangular zone input (easier than polygons)
    with st.expander("➕ Add Rectangular Exclusion Zone", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            lon_min = st.number_input("Min Longitude", value=-76.48, format="%.4f", key="rect_lon_min")
            lat_min = st.number_input("Min Latitude", value=38.15, format="%.4f", key="rect_lat_min")
        
        with col2:
            lon_max = st.number_input("Max Longitude", value=-76.47, format="%.4f", key="rect_lon_max")
            lat_max = st.number_input("Max Latitude", value=38.16, format="%.4f", key="rect_lat_max")
        
        if st.button("Add Rectangle", type="primary"):
            # Create rectangular polygon
            coords = [
                [lon_min, lat_min],
                [lon_max, lat_min],
                [lon_max, lat_max],
                [lon_min, lat_max],
                [lon_min, lat_min]  # Close the polygon
            ]
            st.session_state.exclusion_polygons.append({
                'type': 'polygon',
                'coordinates': coords,
                'name': 'Rectangle'
            })
            save_exclusion_zones(st.session_state.exclusion_polygons)
            st.success("Added rectangular exclusion zone")
            st.rerun()
    
    # Load water boundary
    water_geometry = None
    try:
        from usgs_data_download import load_water_boundary_data
        from shapely.vectorized import contains
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
    except:
        contains = None
    
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
    
    # Apply polygon exclusion zones EFFICIENTLY
    for zone in st.session_state.exclusion_polygons:
        if zone.get('type') == 'polygon' and len(zone.get('coordinates', [])) >= 3:
            # Use vectorized point-in-polygon test
            inside_polygon = points_in_polygon(lon_flat, lat_flat, zone['coordinates'])
            mask = mask & ~inside_polygon  # Exclude points inside polygon
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Create the map
    fig = go.Figure()
    
    # Add settlement probability
    if len(prob_filtered) > 0:
        prob_transformed = np.sqrt(prob_filtered)
        p25 = np.percentile(prob_transformed, 25)
        p75 = np.percentile(prob_transformed, 75)
        iqr = p75 - p25
        color_min = max(0, p25 - 0.5 * iqr)
        color_max = min(1, p75 + 0.5 * iqr)
    else:
        prob_transformed = prob_filtered
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
            colorbar=dict(title="Probability")
        ),
        name='Settlement',
        hovertemplate='Lon: %{lon:.4f}<br>Lat: %{lat:.4f}<br>Prob: %{customdata:.3f}<extra></extra>',
        customdata=prob_filtered
    ))
    
    # Add exclusion polygons
    for i, zone in enumerate(st.session_state.exclusion_polygons):
        if zone.get('type') == 'polygon':
            coords = zone['coordinates']
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            fig.add_trace(go.Scattermapbox(
                lon=lons,
                lat=lats,
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
        mode='markers',
        marker=dict(size=10, color='white'),
        text=reef_data['SourceReef'],
        name='Reefs',
        hovertemplate='%{text}<extra></extra>'
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
    
    # Show current zones
    if st.session_state.exclusion_polygons:
        with st.expander(f"📍 {len(st.session_state.exclusion_polygons)} Exclusion Zones"):
            for i, zone in enumerate(st.session_state.exclusion_polygons):
                col1, col2 = st.columns([5, 1])
                with col1:
                    n_vertices = len(zone['coordinates']) - 1
                    st.text(f"Zone {i+1}: {zone.get('name', 'Polygon')} ({n_vertices} vertices)")
                with col2:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state.exclusion_polygons.pop(i)
                        save_exclusion_zones(st.session_state.exclusion_polygons)
                        st.rerun()
    
    st.info("🌊 Land areas are automatically filtered using water boundary data")
    st.info(f"📁 Zones saved to: {EXCLUSION_ZONES_FILE}")