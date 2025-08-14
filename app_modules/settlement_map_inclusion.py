"""
Settlement Map with Inclusion Zones
Only shows settlement INSIDE the drawn polygons (water areas)
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import json
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# File to store inclusion zones
INCLUSION_ZONES_FILE = 'data/inclusion_zones.json'

def load_inclusion_zones():
    """Load saved inclusion zones from file"""
    if os.path.exists(INCLUSION_ZONES_FILE):
        try:
            with open(INCLUSION_ZONES_FILE, 'r') as f:
                data = json.load(f)
                return data.get('zones', [])
        except:
            return []
    return []

def points_in_polygon(points_x, points_y, poly_coords):
    """Vectorized point-in-polygon test using ray casting"""
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
    """Render settlement map with inclusion zones"""
    st.header("🎯 Settlement Map with Inclusion Zones")
    
    # Load inclusion zones
    inclusion_zones = load_inclusion_zones()
    
    if inclusion_zones:
        st.success(f"✅ Using {len(inclusion_zones)} inclusion zones from drawing tool")
        st.info("Settlement is calculated only INSIDE the drawn water areas")
    else:
        st.warning("⚠️ No inclusion zones found. Run `python3 draw_inclusion_zones.py` to draw water areas.")
    
    # Button to reload zones
    if st.button("🔄 Reload Inclusion Zones"):
        st.rerun()
    
    # Load reef data
    n_reefs = min(27, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
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
    
    # Start with basic threshold
    mask = prob_flat > 0.01
    
    # INCLUSION ZONES - Only keep points INSIDE the polygons
    if inclusion_zones:
        # Start with all points masked out
        inclusion_mask = np.zeros(len(lon_flat), dtype=bool)
        
        # Add points that are inside ANY inclusion zone
        for zone in inclusion_zones:
            if zone.get('type') == 'inclusion' and len(zone.get('coordinates', [])) >= 3:
                inside_this_polygon = points_in_polygon(lon_flat, lat_flat, zone['coordinates'])
                inclusion_mask = inclusion_mask | inside_this_polygon  # Union of all inclusion zones
        
        # Apply inclusion mask
        mask = mask & inclusion_mask
        
        if not np.any(mask):
            st.warning("No settlement points found inside inclusion zones. Check zone boundaries.")
    
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
        hovertemplate='Lon: %{lon:.5f}<br>Lat: %{lat:.5f}<br>Prob: %{customdata:.3f}<extra></extra>',
        customdata=prob_filtered
    ))
    
    # Add inclusion zone outlines
    if inclusion_zones:
        for i, zone in enumerate(inclusion_zones):
            if zone.get('type') == 'inclusion':
                coords = zone['coordinates']
                lons = [c[0] for c in coords]
                lats = [c[1] for c in coords]
                
                # Draw zone boundary
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(color='blue', width=2),
                    name=f'Water Zone {i+1}',
                    hoverinfo='name'
                ))
    
    # Add reefs
    fig.add_trace(go.Scattermapbox(
        lon=reef_data['Longitude'],
        lat=reef_data['Latitude'],
        mode='markers+text',
        marker=dict(size=10, color='white', opacity=1),
        text=reef_data['SourceReef'],
        textfont=dict(size=8, color='black'),
        textposition="top center",
        name='Reefs',
        hovertemplate='<b>%{text}</b><extra></extra>'
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
        title="Settlement Probability (Inside Water Areas Only)"
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Instructions
    with st.expander("📖 How to Define Water Areas"):
        st.markdown("""
        1. **Run the drawing tool**: `python3 draw_inclusion_zones.py`
        2. **Draw polygons** around water areas where settlement should be calculated
        3. **Save** the polygons (press 's' in the drawing tool)
        4. **Reload** this page to see the updated settlement map
        
        **Controls in drawing tool:**
        - Left-click: Add vertex
        - Right-click: Close polygon
        - 's': Save all polygons
        - 'c': Clear current polygon
        - 'u': Undo last point
        - 'd': Delete last polygon
        - 'q': Quit
        
        The blue lines show the boundaries of water areas where settlement is calculated.
        """)
    
    # Show statistics
    if len(prob_filtered) > 0:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Points in water", len(prob_filtered))
        with col2:
            st.metric("Mean probability", f"{np.mean(prob_filtered):.4f}")
        with col3:
            st.metric("Max probability", f"{np.max(prob_filtered):.4f}")
    
    st.info(f"📁 Inclusion zones file: {INCLUSION_ZONES_FILE}")