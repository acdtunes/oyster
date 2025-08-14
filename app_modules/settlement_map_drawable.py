"""
Settlement Map with Drawable Polygon Exclusion Zones
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
    from shapely.geometry import Polygon, Point
    from shapely.vectorized import contains
    SHAPELY_AVAILABLE = True
except ImportError:
    SHAPELY_AVAILABLE = False
    st.error("Shapely is required for this feature. Please install it with: pip install shapely")

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

def render_section(reef_metrics):
    """Render settlement map with drawable exclusion zones"""
    st.header("🎨 Settlement Map with Drawable Exclusion Zones")
    
    if not SHAPELY_AVAILABLE:
        st.error("This module requires Shapely. Please install it.")
        return
    
    # Initialize session state
    if 'exclusion_polygons' not in st.session_state:
        st.session_state.exclusion_polygons = load_exclusion_zones()
    
    if 'drawing_mode' not in st.session_state:
        st.session_state.drawing_mode = False
    
    # Load reef data
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("✏️ Draw Mode", type="primary" if not st.session_state.drawing_mode else "secondary"):
            st.session_state.drawing_mode = not st.session_state.drawing_mode
            st.rerun()
    
    with col2:
        if st.button("💾 Save Zones"):
            save_exclusion_zones(st.session_state.exclusion_polygons)
            st.success(f"Saved {len(st.session_state.exclusion_polygons)} zones")
    
    with col3:
        if st.button("🔄 Reload"):
            st.session_state.exclusion_polygons = load_exclusion_zones()
            st.rerun()
    
    with col4:
        if st.button("🗑️ Clear All"):
            st.session_state.exclusion_polygons = []
            save_exclusion_zones([])
            st.rerun()
    
    # Drawing mode - simplified polygon input
    if st.session_state.drawing_mode:
        st.info("🎨 **Drawing Mode Active** - Define a polygon by entering coordinates")
        
        with st.form("polygon_form"):
            st.markdown("### Define Exclusion Polygon")
            st.markdown("Enter coordinates for each vertex of the polygon (minimum 3 points)")
            
            # Text area for polygon coordinates
            coords_text = st.text_area(
                "Polygon Coordinates (one per line, format: lon,lat)",
                value="-76.475,38.18\n-76.470,38.185\n-76.465,38.18\n-76.470,38.175",
                height=150
            )
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("Add Polygon", type="primary"):
                    try:
                        # Parse coordinates
                        lines = coords_text.strip().split('\n')
                        coords = []
                        for line in lines:
                            if ',' in line:
                                lon, lat = line.split(',')
                                coords.append([float(lon.strip()), float(lat.strip())])
                        
                        if len(coords) >= 3:
                            # Close the polygon if not already closed
                            if coords[0] != coords[-1]:
                                coords.append(coords[0])
                            
                            # Add to exclusion zones
                            st.session_state.exclusion_polygons.append({
                                'type': 'polygon',
                                'coordinates': coords
                            })
                            save_exclusion_zones(st.session_state.exclusion_polygons)
                            st.success(f"Added polygon with {len(coords)-1} vertices")
                            st.session_state.drawing_mode = False
                            st.rerun()
                        else:
                            st.error("Need at least 3 points for a polygon")
                    except Exception as e:
                        st.error(f"Error parsing coordinates: {e}")
            
            with col2:
                if st.form_submit_button("Cancel"):
                    st.session_state.drawing_mode = False
                    st.rerun()
        
        st.markdown("### Quick Templates")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Add Small Circle"):
                center_lon = -76.47
                center_lat = 38.18
                radius = 0.005
                n_points = 20
                angles = np.linspace(0, 2*np.pi, n_points)
                coords = [[center_lon + radius*np.cos(a), center_lat + radius*np.sin(a)] 
                         for a in angles]
                coords.append(coords[0])  # Close the polygon
                
                st.session_state.exclusion_polygons.append({
                    'type': 'polygon',
                    'coordinates': coords
                })
                save_exclusion_zones(st.session_state.exclusion_polygons)
                st.success("Added circular exclusion zone")
                st.rerun()
        
        with col2:
            if st.button("Add Rectangle"):
                coords = [
                    [-76.475, 38.175],
                    [-76.465, 38.175],
                    [-76.465, 38.185],
                    [-76.475, 38.185],
                    [-76.475, 38.175]
                ]
                st.session_state.exclusion_polygons.append({
                    'type': 'polygon',
                    'coordinates': coords
                })
                save_exclusion_zones(st.session_state.exclusion_polygons)
                st.success("Added rectangular exclusion zone")
                st.rerun()
    
    # Load water boundary
    water_geometry = None
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
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
    if water_geometry is not None:
        water_mask = contains(water_geometry, lon_flat, lat_flat)
        mask = mask & water_mask
    
    # Apply polygon exclusion zones
    for zone in st.session_state.exclusion_polygons:
        if zone['type'] == 'polygon' and len(zone['coordinates']) >= 3:
            polygon = Polygon(zone['coordinates'])
            # Check each point
            for i in range(len(lon_flat)):
                if mask[i] and polygon.contains(Point(lon_flat[i], lat_flat[i])):
                    mask[i] = False
    
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
            colorbar=dict(title="Probability", thickness=20)
        ),
        name='Settlement',
        hovertemplate='%{customdata:.3f}<extra></extra>',
        customdata=prob_filtered
    ))
    
    # Add exclusion polygons
    for i, zone in enumerate(st.session_state.exclusion_polygons):
        if zone['type'] == 'polygon':
            coords = zone['coordinates']
            lons = [c[0] for c in coords]
            lats = [c[1] for c in coords]
            
            # Draw polygon
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
    
    # Show current zones list
    if st.session_state.exclusion_polygons:
        with st.expander(f"📍 {len(st.session_state.exclusion_polygons)} Exclusion Zones", expanded=False):
            for i, zone in enumerate(st.session_state.exclusion_polygons):
                col1, col2 = st.columns([5, 1])
                with col1:
                    n_vertices = len(zone['coordinates']) - 1 if zone['coordinates'][0] == zone['coordinates'][-1] else len(zone['coordinates'])
                    st.text(f"Zone {i+1}: Polygon with {n_vertices} vertices")
                with col2:
                    if st.button("❌", key=f"del_{i}", help=f"Delete zone {i+1}"):
                        st.session_state.exclusion_polygons.pop(i)
                        save_exclusion_zones(st.session_state.exclusion_polygons)
                        st.rerun()
    
    # Info
    st.info("🌊 Land areas are automatically filtered using water boundary data")
    st.info(f"📁 Exclusion zones saved to: {EXCLUSION_ZONES_FILE}")