"""
Settlement Map visualization module
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
import sys
import os
import json
sys.path.append('.')
from shapely.vectorized import contains

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

def create_visualization(reef_metrics):
    """Create settlement probability map with current-based dispersal"""
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    try:
        from python_dispersal_model import calculate_advection_diffusion_settlement
        
        # Calculate current-based settlement field
        # The bounds are now set directly in the dispersal model
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
    
    # Load water boundary using the proper function
    try:
        from usgs_data_download import load_water_boundary_data
        
        # Load the cached water boundary geometry
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
        
        if water_geometry is not None:
            # Info message (only show once)
            if 'water_boundary_loaded' not in st.session_state:
                st.success("✅ Using St. Mary's River water boundary")
                st.session_state.water_boundary_loaded = True
        else:
            st.warning("Water boundary not found - showing all points")
            
    except Exception as e:
        st.warning(f"Could not load water boundary: {e}")
        water_geometry = None
    
    # Flatten arrays for scattermapbox
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    prob_flat = settlement_prob.flatten()
    
    # Filter to keep only water points and meaningful probabilities
    threshold = 0.01
    
    if water_geometry is not None:
        # Vectorized water mask evaluation using shapely
        water_mask_flat = contains(water_geometry, lon_flat, lat_flat)
        mask = (prob_flat > threshold) & water_mask_flat
    else:
        # No water mask available, just use probability threshold
        mask = prob_flat > threshold
    
    # Apply saved inclusion zones - ONLY show inside water polygons
    inclusion_zones = load_inclusion_zones()
    
    # Efficient vectorized point-in-polygon function
    def points_in_polygon_vec(points_x, points_y, poly_coords):
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
    
    # If we have inclusion zones, only keep points INSIDE them
    if inclusion_zones:
        inclusion_mask = np.zeros(len(lon_flat), dtype=bool)
        
        for zone in inclusion_zones:
            if zone.get('type') == 'inclusion' and len(zone.get('coordinates', [])) >= 3:
                inside_polygon = points_in_polygon_vec(lon_flat, lat_flat, zone['coordinates'])
                inclusion_mask = inclusion_mask | inside_polygon  # Union of all zones
        
        mask = mask & inclusion_mask
    
    lon_filtered = lon_flat[mask]
    lat_filtered = lat_flat[mask]
    prob_filtered = prob_flat[mask]
    
    # Use power transformation to enhance contrast
    # Square root expands lower values while keeping range 0-1
    prob_transformed = np.sqrt(prob_filtered)
    
    # Calculate percentiles for better color scaling
    if len(prob_transformed) > 0:
        # Use interquartile range for color scale
        p25 = np.percentile(prob_transformed, 25)
        p75 = np.percentile(prob_transformed, 75)
        
        # Expand the range slightly for better contrast
        iqr = p75 - p25
        color_min = max(0, p25 - 0.5 * iqr)
        color_max = min(1, p75 + 0.5 * iqr)
        color_values = prob_transformed
        
        # Log the range for debugging
        st.info(f"Color scale: {color_min:.3f} to {color_max:.3f} | Data range: {prob_filtered.min():.3f} to {prob_filtered.max():.3f}")
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
            color=color_values,  # Use log-transformed values
            colorscale='Turbo',  # Better contrast than YlOrRd
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
        customdata=prob_filtered,  # Show original probabilities in hover
        name='Settlement Field'
    ))
    
    # Note: Inclusion zones are applied to filter data but not displayed as borders
    
    # Create color mapping for reef types
    def get_reef_color(reef_type):
        color_map = {
            'Source': '#4CAF50',    # Green - net larval exporters
            'Sink': '#2196F3',      # Blue - net larval importers
            'Balanced': '#FF9800'   # Orange - equal import/export
        }
        return color_map.get(reef_type, '#757575')  # Gray for unknown
    
    # Scale density values to reasonable marker sizes (6-20 px range)
    min_density = reef_data['Density'].min()
    max_density = reef_data['Density'].max()
    
    def scale_density_to_size(density):
        # Scale density to marker size between 6 and 20 pixels
        if max_density > min_density:
            normalized = (density - min_density) / (max_density - min_density)
            return 6 + (normalized * 14)  # 6-20 px range
        else:
            return 10  # Default size if all densities are the same
    
    # Add reef locations by type for proper legend
    reef_types = ['Source', 'Sink', 'Balanced']
    type_colors = {'Source': '#4CAF50', 'Sink': '#2196F3', 'Balanced': '#FF9800'}
    
    for reef_type in reef_types:
        type_data = reef_data[reef_data['Type'] == reef_type]
        if len(type_data) > 0:
            type_sizes = [scale_density_to_size(density) for density in type_data['Density']]
            
            fig.add_trace(go.Scattermapbox(
                lon=type_data['Longitude'],
                lat=type_data['Latitude'],
                mode='markers',
                marker=dict(
                    size=type_sizes,
                    color=type_colors[reef_type],
                    opacity=1
                ),
                hovertemplate='<b>%{customdata[0]}</b><br>Type: ' + reef_type + '<br>Density: %{customdata[1]:.1f}<br>Lon: %{lon:.3f}<br>Lat: %{lat:.3f}<extra></extra>',
                customdata=list(zip(type_data['SourceReef'], type_data['Density'])),
                name=f'{reef_type} Reefs',
                showlegend=True
            ))
    
    # Add any unknown/other types
    other_data = reef_data[~reef_data['Type'].isin(reef_types)]
    if len(other_data) > 0:
        other_sizes = [scale_density_to_size(density) for density in other_data['Density']]
        
        fig.add_trace(go.Scattermapbox(
            lon=other_data['Longitude'],
            lat=other_data['Latitude'],
            mode='markers',
            marker=dict(
                size=other_sizes,
                color='#757575',
                opacity=1
            ),
            hovertemplate='<b>%{customdata[0]}</b><br>Type: %{customdata[1]}<br>Density: %{customdata[2]:.1f}<br>Lon: %{lon:.3f}<br>Lat: %{lat:.3f}<extra></extra>',
            customdata=list(zip(other_data['SourceReef'], other_data['Type'], other_data['Density'])),
            name='Other Reefs',
            showlegend=True
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
            bgcolor="rgba(255,255,255,0.95)",
            bordercolor="rgba(0,0,0,0.5)",
            borderwidth=2,
            font=dict(size=12, color="black")
        )
    )
    
    return fig

def render_section(reef_metrics):
    """Render the complete settlement map section"""
    st.header("Settlement Probability Map")
    st.markdown("""
    Areas with higher settlement probability (warmer colors) indicate favorable zones for larval settlement.
    
    **Reef markers are color-coded by type and sized by density:**
    - **Color**: Green (Source), Blue (Sink), Orange (Balanced)
    - **Size**: Proportional to reef density (larger = higher density)
    - **Legend**: Shown on left side of map
    """)
    
    # Show if inclusion zones are being applied
    inclusion_zones = load_inclusion_zones()
    if inclusion_zones:
        st.info(f"🌊 Using {len(inclusion_zones)} water inclusion zones. Settlement shown only inside water areas.")
    
    fig = create_visualization(reef_metrics)
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    
    st.info("Settlement probability is calculated using an advection-diffusion model with ocean currents.")