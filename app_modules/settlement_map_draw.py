"""
Settlement Map with Drawable Polygon Exclusion Zones using Plotly
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json
import os
import sys
from streamlit_plotly_events import plotly_events

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
    """Vectorized point-in-polygon test"""
    poly_x = np.array([c[0] for c in poly_coords])
    poly_y = np.array([c[1] for c in poly_coords])
    n = len(poly_coords) - 1
    
    inside = np.zeros(len(points_x), dtype=bool)
    
    for i in range(n):
        j = (i + 1) % n
        xi, yi = poly_x[i], poly_y[i]
        xj, yj = poly_x[j], poly_y[j]
        
        intersect = ((yi > points_y) != (yj > points_y)) & \
                   (points_x < (xj - xi) * (points_y - yi) / (yj - yi) + xi)
        inside = np.logical_xor(inside, intersect)
    
    return inside

def render_section(reef_metrics):
    """Render settlement map with drawable exclusion zones"""
    st.header("🎨 Interactive Settlement Map - Draw Exclusion Zones")
    
    # Initialize session state
    if 'exclusion_polygons' not in st.session_state:
        st.session_state.exclusion_polygons = load_exclusion_zones()
    
    if 'drawing_polygon' not in st.session_state:
        st.session_state.drawing_polygon = []
    
    # Instructions
    st.info("""
    **How to draw exclusion zones:**
    1. Click points on the map to create a polygon
    2. Click "Close Polygon" when done
    3. The area inside will be excluded from settlement calculations
    4. Click "Save Zones" to persist your changes
    """)
    
    # Load reef data
    n_reefs = min(27, len(reef_metrics))  # Use 27 to match matrix
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Control buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("🔄 Clear Drawing"):
            st.session_state.drawing_polygon = []
            st.rerun()
    
    with col2:
        if st.button("✅ Close Polygon"):
            if len(st.session_state.drawing_polygon) >= 3:
                # Close the polygon
                polygon = st.session_state.drawing_polygon.copy()
                if polygon[0] != polygon[-1]:
                    polygon.append(polygon[0])
                
                st.session_state.exclusion_polygons.append({
                    'type': 'polygon',
                    'coordinates': polygon
                })
                st.session_state.drawing_polygon = []
                save_exclusion_zones(st.session_state.exclusion_polygons)
                st.success("Added polygon!")
                st.rerun()
            else:
                st.error("Need at least 3 points for a polygon")
    
    with col3:
        if st.button("💾 Save All"):
            save_exclusion_zones(st.session_state.exclusion_polygons)
            st.success(f"Saved {len(st.session_state.exclusion_polygons)} zones")
    
    with col4:
        if st.button("🗑️ Clear All"):
            st.session_state.exclusion_polygons = []
            st.session_state.drawing_polygon = []
            save_exclusion_zones([])
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
    
    # Apply polygon exclusion zones
    for zone in st.session_state.exclusion_polygons:
        if zone.get('type') == 'polygon' and len(zone.get('coordinates', [])) >= 3:
            inside_polygon = points_in_polygon(lon_flat, lat_flat, zone['coordinates'])
            mask = mask & ~inside_polygon
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Create the interactive map
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
        name='Settlement'
    ))
    
    # Add existing exclusion polygons
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
                name=f'Zone {i+1}'
            ))
    
    # Add current drawing polygon
    if st.session_state.drawing_polygon:
        coords = st.session_state.drawing_polygon
        lons = [c[0] for c in coords]
        lats = [c[1] for c in coords]
        
        # Add lines
        fig.add_trace(go.Scattermapbox(
            lon=lons,
            lat=lats,
            mode='lines+markers',
            line=dict(color='yellow', width=3),
            marker=dict(size=8, color='yellow'),
            name='Drawing'
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
        showlegend=True,
        clickmode='event'
    )
    
    # Create a placeholder for the map
    map_placeholder = st.empty()
    
    # Render the map and capture clicks
    with map_placeholder.container():
        # Use plotly chart with key to capture events
        clicked_point = st.plotly_chart(
            fig, 
            use_container_width=True, 
            key="map_click",
            on_select="rerun",
            selection_mode="points"
        )
    
    # Alternative: Add manual coordinate input for polygon vertices
    with st.expander("📍 Manual Polygon Input (Alternative Method)"):
        st.markdown("Click points on the map above, or enter coordinates manually:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            lon_input = st.number_input("Longitude", value=-76.47, format="%.4f", key="lon_input")
        with col2:
            lat_input = st.number_input("Latitude", value=38.18, format="%.4f", key="lat_input")
        with col3:
            if st.button("Add Point"):
                st.session_state.drawing_polygon.append([lon_input, lat_input])
                st.rerun()
        
        if st.session_state.drawing_polygon:
            st.write("Current polygon points:")
            for i, point in enumerate(st.session_state.drawing_polygon):
                st.text(f"{i+1}. [{point[0]:.4f}, {point[1]:.4f}]")
    
    # Show saved zones
    if st.session_state.exclusion_polygons:
        with st.expander(f"📍 {len(st.session_state.exclusion_polygons)} Saved Zones"):
            for i, zone in enumerate(st.session_state.exclusion_polygons):
                col1, col2 = st.columns([5, 1])
                with col1:
                    n_vertices = len(zone['coordinates']) - 1
                    st.text(f"Zone {i+1}: Polygon with {n_vertices} vertices")
                with col2:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state.exclusion_polygons.pop(i)
                        save_exclusion_zones(st.session_state.exclusion_polygons)
                        st.rerun()
    
    st.info("🌊 Land areas are automatically filtered using water boundary data")
    st.info(f"📁 Zones saved to: {EXCLUSION_ZONES_FILE}")