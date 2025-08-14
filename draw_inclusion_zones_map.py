#!/usr/bin/env python3
"""
Desktop application for drawing inclusion zones on St. Mary's River map
Click to add vertices, right-click to close polygon
The polygons define areas WHERE settlement should be calculated (water areas)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
import json
import pandas as pd
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import math

class PolygonDrawerMap:
    def __init__(self):
        # Load reef data to get map bounds
        self.reef_data = pd.read_csv('output/st_marys/reef_metrics.csv')
        
        # Set map bounds for St. Mary's River
        self.lon_min = -76.495
        self.lon_max = -76.4
        self.lat_min = 38.125
        self.lat_max = 38.23
        
        # Storage for polygons
        self.current_polygon = []
        self.saved_polygons = []
        
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(14, 12))
        
        # Load and display basemap
        self.load_basemap()
        
        # Set up the map
        self.setup_map()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Drawing state
        self.drawing = False
        self.temp_lines = []
        self.temp_points = []
        
    def latlon_to_tile(self, lat, lon, zoom):
        """Convert lat/lon to tile coordinates"""
        lat_rad = math.radians(lat)
        n = 2.0 ** zoom
        x = int((lon + 180.0) / 360.0 * n)
        y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return x, y
    
    def tile_to_latlon(self, x, y, zoom):
        """Convert tile coordinates to lat/lon for corners"""
        n = 2.0 ** zoom
        lon_min = x / n * 360.0 - 180.0
        lon_max = (x + 1) / n * 360.0 - 180.0
        lat_rad_min = math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n)))
        lat_rad_max = math.atan(math.sinh(math.pi * (1 - 2 * y / n)))
        lat_min = math.degrees(lat_rad_min)
        lat_max = math.degrees(lat_rad_max)
        return lon_min, lon_max, lat_min, lat_max
    
    def load_basemap(self):
        """Load OpenStreetMap tiles as basemap"""
        print("Loading basemap tiles...")
        
        # Use a zoom level that gives good detail
        zoom = 13
        
        # Get tile bounds
        x_min, y_max = self.latlon_to_tile(self.lat_min, self.lon_min, zoom)
        x_max, y_min = self.latlon_to_tile(self.lat_max, self.lon_max, zoom)
        
        # Create empty image to hold all tiles
        tile_size = 256
        img_width = (x_max - x_min + 1) * tile_size
        img_height = (y_max - y_min + 1) * tile_size
        
        # Try to load cached basemap first
        cache_file = 'data/st_marys_basemap.png'
        cache_bounds_file = 'data/st_marys_basemap_bounds.json'
        
        try:
            # Load cached image
            print("Loading cached basemap...")
            full_image = Image.open(cache_file)
            with open(cache_bounds_file, 'r') as f:
                bounds = json.load(f)
            lon_min_img = bounds['lon_min']
            lon_max_img = bounds['lon_max']
            lat_min_img = bounds['lat_min']
            lat_max_img = bounds['lat_max']
            
        except:
            print("Downloading basemap tiles (this may take a moment)...")
            full_image = Image.new('RGB', (img_width, img_height))
            
            # Download each tile
            for x in range(x_min, x_max + 1):
                for y in range(y_min, y_max + 1):
                    try:
                        # OpenStreetMap tile URL
                        url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
                        
                        # Add headers to be polite to OSM servers
                        headers = {
                            'User-Agent': 'OysterLarvalAnalysis/1.0'
                        }
                        
                        response = requests.get(url, headers=headers, timeout=10)
                        tile_img = Image.open(BytesIO(response.content))
                        
                        # Paste tile into full image
                        full_image.paste(tile_img, 
                                       ((x - x_min) * tile_size, 
                                        (y - y_min) * tile_size))
                        print(f"Loaded tile {x},{y}")
                        
                    except Exception as e:
                        print(f"Could not load tile {x},{y}: {e}")
                        # Create a gray tile as fallback
                        gray_tile = Image.new('RGB', (tile_size, tile_size), (200, 200, 200))
                        full_image.paste(gray_tile, 
                                       ((x - x_min) * tile_size, 
                                        (y - y_min) * tile_size))
            
            # Calculate the actual bounds of the image
            lon_min_img, _, _, lat_max_img = self.tile_to_latlon(x_min, y_min, zoom)
            _, lon_max_img, lat_min_img, _ = self.tile_to_latlon(x_max, y_max, zoom)
            
            # Save cache
            print("Saving basemap cache...")
            full_image.save(cache_file)
            with open(cache_bounds_file, 'w') as f:
                json.dump({
                    'lon_min': lon_min_img,
                    'lon_max': lon_max_img,
                    'lat_min': lat_min_img,
                    'lat_max': lat_max_img
                }, f)
        
        # Display the basemap
        self.ax.imshow(full_image, extent=[lon_min_img, lon_max_img, lat_min_img, lat_max_img], 
                      aspect='auto', alpha=0.8, zorder=0)
        
        print("Basemap loaded!")
        
    def setup_map(self):
        """Set up the map with proper coordinates"""
        # Set limits
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        
        # Labels
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title('Draw Water Inclusion Zones on St. Mary\'s River', fontsize=14, fontweight='bold')
        
        # Grid
        self.ax.grid(True, alpha=0.3, color='white', linewidth=0.5)
        
        # Plot reef locations as reference
        reef_subset = self.reef_data.iloc[:27]  # First 27 reefs
        self.ax.scatter(reef_subset['Longitude'], reef_subset['Latitude'], 
                       c='red', s=100, marker='o', label='Reef Sites', 
                       zorder=5, edgecolors='white', linewidth=2)
        
        # Add reef labels with better visibility
        for _, reef in reef_subset.iterrows():
            self.ax.annotate(reef['SourceReef'], 
                           (reef['Longitude'], reef['Latitude']),
                           fontsize=8, ha='center', va='bottom',
                           color='darkred', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.2', 
                                   facecolor='white', alpha=0.7))
        
        # Aspect ratio to match lat/lon scaling
        self.ax.set_aspect(1/np.cos(np.radians(self.lat_min + self.lat_max)/2))
        
        # Legend
        self.ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
        
        # Instructions
        self.add_instructions()
        
    def add_instructions(self):
        """Add instructions to the plot"""
        instructions = [
            "DRAW WATER AREAS:",
            "• Left-click: Add vertex",
            "• Right-click: Close polygon",
            "• 'c': Clear current",
            "• 's': Save all zones",
            "• 'u': Undo last point",
            "• 'd': Delete last polygon",
            "• 'l': Load existing zones",
            "• 'q': Quit",
            "",
            "Draw around WATER areas",
            "where settlement occurs"
        ]
        
        text = '\n'.join(instructions)
        self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightblue', 
                            alpha=0.9, edgecolor='navy', linewidth=2))
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        # Get coordinates
        lon, lat = event.xdata, event.ydata
        
        if event.button == 1:  # Left click - add point
            self.current_polygon.append([lon, lat])
            
            # Plot the point
            point, = self.ax.plot(lon, lat, 'yo', markersize=10, 
                                 markeredgecolor='blue', markeredgewidth=2)
            self.temp_points.append(point)
            
            # Draw line from previous point
            if len(self.current_polygon) > 1:
                prev_point = self.current_polygon[-2]
                line, = self.ax.plot([prev_point[0], lon], [prev_point[1], lat], 
                                    'b-', linewidth=3, alpha=0.7)
                self.temp_lines.append(line)
            
            # Update status
            print(f"Added point {len(self.current_polygon)}: ({lon:.5f}, {lat:.5f})")
            
        elif event.button == 3:  # Right click - close polygon
            if len(self.current_polygon) >= 3:
                # Close the polygon
                self.current_polygon.append(self.current_polygon[0])
                
                # Draw closing line
                last_point = self.current_polygon[-2]
                first_point = self.current_polygon[0]
                line, = self.ax.plot([last_point[0], first_point[0]], 
                                    [last_point[1], first_point[1]], 
                                    'b-', linewidth=3, alpha=0.7)
                self.temp_lines.append(line)
                
                # Create polygon patch
                polygon = Polygon(self.current_polygon, 
                                closed=True, 
                                alpha=0.3, 
                                facecolor='cyan',
                                edgecolor='blue',
                                linewidth=3)
                self.ax.add_patch(polygon)
                
                # Save the polygon
                self.saved_polygons.append({
                    'type': 'inclusion',
                    'coordinates': self.current_polygon.copy()
                })
                
                print(f"Closed polygon with {len(self.current_polygon)-1} vertices")
                print(f"Total polygons: {len(self.saved_polygons)}")
                
                # Clear current polygon
                self.current_polygon = []
                self.clear_temp_drawings()
            else:
                print("Need at least 3 points to close polygon")
        
        plt.draw()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'c':  # Clear current polygon
            self.current_polygon = []
            self.clear_temp_drawings()
            print("Cleared current polygon")
            plt.draw()
            
        elif event.key == 'u':  # Undo last point
            if self.current_polygon:
                self.current_polygon.pop()
                if self.temp_lines:
                    line = self.temp_lines.pop()
                    line.remove()
                if self.temp_points:
                    point = self.temp_points.pop()
                    point.remove()
                print(f"Removed last point. {len(self.current_polygon)} points remaining")
                plt.draw()
                
        elif event.key == 'd':  # Delete last saved polygon
            if self.saved_polygons:
                self.saved_polygons.pop()
                # Redraw everything
                self.ax.clear()
                self.load_basemap()
                self.setup_map()
                self.redraw_polygons()
                print(f"Deleted last polygon. {len(self.saved_polygons)} polygons remaining")
                plt.draw()
                
        elif event.key == 's':  # Save polygons to file
            self.save_polygons()
            
        elif event.key == 'l':  # Load existing zones
            self.load_existing_zones()
            
        elif event.key == 'q':  # Quit
            plt.close()
    
    def clear_temp_drawings(self):
        """Clear temporary drawing elements"""
        for line in self.temp_lines:
            try:
                line.remove()
            except:
                pass
        self.temp_lines = []
        
        for point in self.temp_points:
            try:
                point.remove()
            except:
                pass
        self.temp_points = []
        
    def redraw_polygons(self):
        """Redraw all saved polygons"""
        for poly_data in self.saved_polygons:
            polygon = Polygon(poly_data['coordinates'], 
                            closed=True, 
                            alpha=0.3, 
                            facecolor='cyan',
                            edgecolor='blue',
                            linewidth=3)
            self.ax.add_patch(polygon)
    
    def save_polygons(self):
        """Save polygons to JSON file"""
        if not self.saved_polygons:
            print("No polygons to save")
            return
        
        filename = 'data/inclusion_zones.json'
        
        # Prepare data
        save_data = {
            'type': 'inclusion_zones',
            'created': datetime.now().isoformat(),
            'bounds': {
                'lon_min': self.lon_min,
                'lon_max': self.lon_max,
                'lat_min': self.lat_min,
                'lat_max': self.lat_max
            },
            'zones': self.saved_polygons
        }
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n✅ Saved {len(self.saved_polygons)} inclusion zones to {filename}")
        
        # Also save a backup with timestamp
        backup_filename = f'data/inclusion_zones_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"📁 Backup saved to {backup_filename}")
    
    def load_existing_zones(self):
        """Load and display existing zones"""
        try:
            with open('data/inclusion_zones.json', 'r') as f:
                data = json.load(f)
                if 'zones' in data:
                    self.saved_polygons = data['zones']
                    self.redraw_polygons()
                    print(f"✅ Loaded {len(self.saved_polygons)} existing zones")
                    plt.draw()
        except FileNotFoundError:
            print("❌ No existing zones found")
        except Exception as e:
            print(f"❌ Error loading zones: {e}")
    
    def run(self):
        """Run the application"""
        # Automatically load existing zones at startup
        self.load_existing_zones()
        
        plt.show()

def main():
    print("="*60)
    print("🦪 WATER INCLUSION ZONE DRAWING TOOL")
    print("="*60)
    print("\nDraw polygons around WATER AREAS where oyster")
    print("larvae can settle. The map shows St. Mary's River.")
    print("\nRed markers show existing reef locations.")
    print("\nStarting application...")
    print("="*60)
    
    app = PolygonDrawerMap()
    app.run()
    
    print("\nApplication closed.")

if __name__ == "__main__":
    main()