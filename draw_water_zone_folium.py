#!/usr/bin/env python3
"""
Real interactive map using Folium for drawing water inclusion zones
Opens in web browser with satellite imagery and proper zoom/pan
"""

import folium
from folium import plugins
import pandas as pd
import json
import webbrowser
import tempfile
import os
from datetime import datetime

def create_interactive_map():
    """Create a Folium map for drawing water inclusion zones"""
    
    # Load reef data
    reef_data = pd.read_csv('output/st_marys/reef_metrics.csv')
    
    # Calculate center of St. Mary's River
    center_lat = 38.18
    center_lon = -76.45
    
    # Create map with satellite imagery
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=13,
        tiles=None  # We'll add custom tiles
    )
    
    # Add satellite imagery
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add OpenStreetMap for reference
    folium.TileLayer(
        tiles='OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # Add reef locations
    reef_subset = reef_data.iloc[:27]
    for _, reef in reef_subset.iterrows():
        folium.CircleMarker(
            location=[reef['Latitude'], reef['Longitude']],
            radius=8,
            popup=reef['SourceReef'],
            color='red',
            fill=True,
            fillColor='red',
            fillOpacity=0.8,
            weight=2
        ).add_to(m)
    
    # Add drawing tools
    draw = plugins.Draw(
        export=True,
        filename='water_inclusion_zone.geojson',
        position='topleft',
        draw_options={
            'polyline': False,
            'rectangle': True,
            'polygon': True,
            'circle': False,
            'marker': False,
            'circlemarker': False,
        },
        edit_options={'edit': True}
    )
    draw.add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Add instructions
    instructions_html = """
    <div style="position: fixed; 
                top: 100px; left: 50px; width: 300px; height: 200px; 
                background-color:white; border:2px solid grey; z-index:9999; 
                font-size:14px; padding: 10px;
                box-shadow: 0 0 15px rgba(0,0,0,0.2);
                border-radius: 5px;
                ">
    <h4>🌊 Draw Water Inclusion Zone</h4>
    <p><b>Instructions:</b></p>
    <ol>
    <li>Use the polygon tool in the top-left toolbar</li>
    <li>Click points around water areas where settlement occurs</li>
    <li>Double-click to finish the polygon</li>
    <li>Click "Export" to download the zone</li>
    <li>Move the downloaded file to the data/ folder</li>
    </ol>
    <p><b>Red dots = Reef locations</b></p>
    <p>Switch between Satellite and Street views using layer control</p>
    </div>
    """
    
    m.get_root().html.add_child(folium.Element(instructions_html))
    
    return m

def main():
    print("="*60)
    print("🗺️ REAL INTERACTIVE MAP - WATER ZONE DRAWING")
    print("="*60)
    print("\nCreating interactive map with satellite imagery...")
    print("This will open in your web browser.")
    print("\n📍 Instructions:")
    print("1. Use polygon drawing tool in top-left")
    print("2. Click around water areas")
    print("3. Double-click to finish")
    print("4. Export the zone as GeoJSON")
    print("5. Save to data/inclusion_zones.geojson")
    print("="*60)
    
    # Create the map
    m = create_interactive_map()
    
    # Save to temporary file and open in browser
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
    temp_file.close()
    
    m.save(temp_file.name)
    
    print(f"\n🌐 Opening map in browser...")
    print(f"📁 Map file: {temp_file.name}")
    
    # Open in default browser
    webbrowser.open('file://' + temp_file.name)
    
    print("\n✨ Map opened! Draw your water inclusion zone.")
    print("💾 Export the polygon and save as 'data/inclusion_zones.geojson'")
    print("\nPress Enter when done...")
    input()
    
    # Convert exported GeoJSON to our format if it exists
    if os.path.exists('water_inclusion_zone.geojson'):
        convert_geojson_to_our_format('water_inclusion_zone.geojson')
    
    # Clean up temp file
    try:
        os.unlink(temp_file.name)
    except:
        pass

def convert_geojson_to_our_format(geojson_file):
    """Convert exported GeoJSON to our inclusion zones format"""
    try:
        with open(geojson_file, 'r') as f:
            geojson_data = json.load(f)
        
        zones = []
        for feature in geojson_data.get('features', []):
            if feature['geometry']['type'] == 'Polygon':
                # Get coordinates (note: GeoJSON is [lon, lat])
                coords = feature['geometry']['coordinates'][0]  # First ring
                # Convert to our format [lon, lat]
                zone_coords = [[coord[0], coord[1]] for coord in coords]
                
                zones.append({
                    'type': 'inclusion',
                    'coordinates': zone_coords
                })
        
        if zones:
            # Save in our format
            os.makedirs('data', exist_ok=True)
            save_data = {
                'type': 'inclusion_zones',
                'created': datetime.now().isoformat(),
                'bounds': {
                    'lon_min': -76.495,
                    'lon_max': -76.4,
                    'lat_min': 38.125,
                    'lat_max': 38.23
                },
                'zones': zones
            }
            
            with open('data/inclusion_zones.json', 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"\n✅ Converted and saved {len(zones)} zones to data/inclusion_zones.json")
            print("🎉 Ready to use in Streamlit app!")
        else:
            print("❌ No polygons found in exported file")
            
    except Exception as e:
        print(f"❌ Error converting file: {e}")
        print("💡 Manually save your polygon as data/inclusion_zones.geojson")

if __name__ == "__main__":
    main()