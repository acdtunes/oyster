#!/usr/bin/env python3
"""
Test script to verify satellite map rendering in Streamlit app
"""

import plotly.graph_objects as go
import numpy as np

def test_satellite_map():
    """Create a simple test map with satellite imagery"""
    
    # St. Mary's River center coordinates
    center_lat = 38.190
    center_lon = -76.440
    
    # Create figure with satellite basemap
    fig = go.Figure()
    
    # Add test points
    fig.add_trace(go.Scattermapbox(
        mode='markers',
        lon=[center_lon],
        lat=[center_lat],
        marker=dict(size=15, color='red'),
        text=['St. Mary\'s River Center'],
        name='Test Point'
    ))
    
    # Set satellite map style
    fig.update_layout(
        mapbox=dict(
            style='satellite',  # Satellite imagery
            center=dict(lat=center_lat, lon=center_lon),
            zoom=12
        ),
        showlegend=False,
        height=600,
        title='Satellite Map Test - St. Mary\'s River'
    )
    
    # Save to HTML for viewing
    fig.write_html('test_satellite_map.html')
    print("✅ Test satellite map saved to test_satellite_map.html")
    print("Map style: satellite")
    print(f"Center: {center_lat}°N, {center_lon}°W")
    print("Open the HTML file in a browser to verify satellite imagery is displayed")

if __name__ == "__main__":
    test_satellite_map()