#!/usr/bin/env python3
"""
Test different map styles that work without Mapbox token
"""

import plotly.graph_objects as go
import numpy as np

def test_map_styles():
    """Test various free map styles"""
    
    # St. Mary's River coordinates
    center_lat = 38.190
    center_lon = -76.440
    
    # Map styles that work without token
    free_styles = [
        'open-street-map',
        'carto-positron', 
        'carto-darkmatter',
        'stamen-terrain',
        'stamen-toner',
        'stamen-watercolor'
    ]
    
    for style in free_styles:
        try:
            fig = go.Figure()
            
            # Add test marker
            fig.add_trace(go.Scattermapbox(
                mode='markers',
                lon=[center_lon],
                lat=[center_lat],
                marker=dict(size=15, color='red'),
                text=[f'Test: {style}'],
            ))
            
            # Set map style
            fig.update_layout(
                mapbox=dict(
                    style=style,
                    center=dict(lat=center_lat, lon=center_lon),
                    zoom=12
                ),
                showlegend=False,
                height=400,
                title=f'Map Style: {style}'
            )
            
            # Save to HTML
            filename = f'test_map_{style.replace("-", "_")}.html'
            fig.write_html(filename)
            print(f"✅ {style}: Saved to {filename}")
            
        except Exception as e:
            print(f"❌ {style}: Failed - {str(e)}")
    
    print("\n📍 Recommended styles for colored maps without token:")
    print("  - open-street-map: Full color street map")
    print("  - stamen-terrain: Topographic with colors")
    print("  - stamen-watercolor: Artistic watercolor style")

if __name__ == "__main__":
    test_map_styles()