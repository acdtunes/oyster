#!/usr/bin/env python3
"""
Interactive map with real tiles, zoom, and pan for drawing inclusion zones
"""

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.widgets import Button
import numpy as np
import json
import pandas as pd
from datetime import datetime
import requests
from PIL import Image
from io import BytesIO
import math
import threading
from concurrent.futures import ThreadPoolExecutor
import os

class InteractiveMapDrawer:
    def __init__(self):
        # Load reef data
        self.reef_data = pd.read_csv('output/st_marys/reef_metrics.csv')
        
        # Initial map bounds for St. Mary's River (zoomed in more)
        self.lon_min = -76.50
        self.lon_max = -76.42
        self.lat_min = 38.13
        self.lat_max = 38.22
        
        # Storage for polygons
        self.current_polygon = []
        self.saved_polygons = []
        self.temp_lines = []
        self.temp_points = []
        
        # Tile cache
        self.tile_cache = {}
        self.current_tiles = []
        
        # View tracking for zoom/pan detection
        self.last_xlim = None
        self.last_ylim = None
        
        # Create figure with navigation toolbar enabled
        plt.rcParams['toolbar'] = 'toolbar2'  # Enable navigation toolbar
        self.fig, self.ax = plt.subplots(figsize=(16, 12))
        
        # Ensure toolbar is visible and enabled
        self.fig.canvas.toolbar_visible = True
        
        # Set up the map
        self.setup_map()
        self.load_tiles()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Auto-reload tiles when view changes (from toolbar zoom/pan)
        self.ax.callbacks.connect('xlim_changed', self.on_view_changed)
        self.ax.callbacks.connect('ylim_changed', self.on_view_changed)
        
        # Also connect to canvas events for more comprehensive detection
        self.fig.canvas.mpl_connect('button_release_event', self.on_mouse_release)
        
        print("Use toolbar to zoom/pan - tiles reload automatically")
        
        # Add custom buttons
        self.create_buttons()
        
    def deg2num(self, lat_deg, lon_deg, zoom):
        """Convert lat/lon to tile numbers"""
        lat_rad = math.radians(lat_deg)
        n = 2.0 ** zoom
        xtile = int((lon_deg + 180.0) / 360.0 * n)
        ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
        return (xtile, ytile)
    
    def num2deg(self, xtile, ytile, zoom):
        """Convert tile numbers to lat/lon"""
        n = 2.0 ** zoom
        lon_deg = xtile / n * 360.0 - 180.0
        lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
        lat_deg = math.degrees(lat_rad)
        return (lat_deg, lon_deg)
    
    def get_tile_bounds(self, xtile, ytile, zoom):
        """Get the lat/lon bounds for a tile"""
        lat_min, lon_min = self.num2deg(xtile, ytile + 1, zoom)
        lat_max, lon_max = self.num2deg(xtile + 1, ytile, zoom)
        return lon_min, lon_max, lat_min, lat_max
    
    def download_tile(self, x, y, zoom):
        """Download a single tile"""
        tile_key = (x, y, zoom)
        
        if tile_key in self.tile_cache:
            return self.tile_cache[tile_key]
        
        try:
            # Use OpenStreetMap street tiles
            url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
            
            headers = {'User-Agent': 'OysterLarvalAnalysis/1.0'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                tile_img = Image.open(BytesIO(response.content))
                self.tile_cache[tile_key] = tile_img
                return tile_img
            else:
                return None
                
        except Exception as e:
            print(f"Failed to download tile {x},{y},{zoom}: {e}")
            return None
    
    def load_tiles(self):
        """Load tiles for current view"""
        # Clear existing tiles
        for artist in self.current_tiles:
            artist.remove()
        self.current_tiles.clear()
        
        # Get current view bounds
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Determine appropriate zoom level based on view extent (higher resolution)
        view_width = xlim[1] - xlim[0]
        if view_width > 0.1:
            zoom = 13  # Higher than before
        elif view_width > 0.05:
            zoom = 14
        elif view_width > 0.025:
            zoom = 15
        elif view_width > 0.0125:
            zoom = 16
        else:
            zoom = 17  # Very high resolution
        
        # Get tile range for current view
        x_min, y_max = self.deg2num(ylim[0], xlim[0], zoom)
        x_max, y_min = self.deg2num(ylim[1], xlim[1], zoom)
        
        print(f"Loading tiles at zoom {zoom}: x={x_min}-{x_max}, y={y_min}-{y_max}")
        
        # Download tiles in parallel
        tile_coords = []
        for x in range(max(0, x_min), min(2**zoom, x_max + 1)):
            for y in range(max(0, y_min), min(2**zoom, y_max + 1)):
                tile_coords.append((x, y, zoom))
        
        # Limit number of tiles to prevent overwhelming
        if len(tile_coords) > 50:
            print(f"Too many tiles ({len(tile_coords)}), reducing zoom")
            zoom = max(10, zoom - 1)
            x_min, y_max = self.deg2num(ylim[0], xlim[0], zoom)
            x_max, y_min = self.deg2num(ylim[1], xlim[1], zoom)
            tile_coords = []
            for x in range(max(0, x_min), min(2**zoom, x_max + 1)):
                for y in range(max(0, y_min), min(2**zoom, y_max + 1)):
                    tile_coords.append((x, y, zoom))
        
        # Download tiles
        with ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(lambda coords: (coords, self.download_tile(*coords)), tile_coords))
        
        # Display tiles
        for (x, y, z), tile_img in results:
            if tile_img is not None:
                lon_min, lon_max, lat_min, lat_max = self.get_tile_bounds(x, y, z)
                
                im = self.ax.imshow(tile_img, 
                                   extent=[lon_min, lon_max, lat_min, lat_max],
                                   aspect='auto', 
                                   zorder=0,
                                   alpha=0.8)
                self.current_tiles.append(im)
        
        # Redraw polygons on top
        self.redraw_all()
        
        # Force redraw
        self.fig.canvas.draw()
        
    def setup_map(self):
        """Set up the map"""
        # Set initial limits
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        
        # Labels and title
        self.ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        self.ax.set_title('Street Map Drawing Tool - Use Navigation Toolbar Below', 
                         fontsize=16, fontweight='bold')
        
        # Grid
        self.ax.grid(True, alpha=0.3, color='white', linewidth=1)
        
        # Plot reef locations (no labels to keep map clean)
        reef_subset = self.reef_data.iloc[:27]
        self.reef_scatter = self.ax.scatter(reef_subset['Longitude'], reef_subset['Latitude'], 
                                          c='red', s=100, marker='o', 
                                          label='Reef Sites', zorder=10, 
                                          edgecolors='white', linewidth=1)
        
        # No reef labels for cleaner map
        self.reef_texts = []
        
        # Aspect ratio
        self.ax.set_aspect(1/np.cos(np.radians((self.lat_min + self.lat_max)/2)))
        
        # Legend
        legend = self.ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        legend.set_zorder(15)
        
    def create_buttons(self):
        """Create custom control buttons"""
        # Create button axes
        ax_save = plt.axes([0.02, 0.02, 0.08, 0.04])
        ax_clear = plt.axes([0.11, 0.02, 0.08, 0.04])
        ax_undo = plt.axes([0.20, 0.02, 0.08, 0.04])
        ax_refresh = plt.axes([0.29, 0.02, 0.08, 0.04])
        ax_zoom_in = plt.axes([0.38, 0.02, 0.06, 0.04])
        ax_zoom_out = plt.axes([0.45, 0.02, 0.06, 0.04])
        
        # Create buttons
        self.btn_save = Button(ax_save, 'Save')
        self.btn_clear = Button(ax_clear, 'Clear')
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_refresh = Button(ax_refresh, 'Refresh')
        self.btn_zoom_in = Button(ax_zoom_in, '+')
        self.btn_zoom_out = Button(ax_zoom_out, '-')
        
        # Connect button callbacks
        self.btn_save.on_clicked(lambda x: self.save_polygons())
        self.btn_clear.on_clicked(lambda x: self.clear_current())
        self.btn_undo.on_clicked(lambda x: self.undo_point())
        self.btn_refresh.on_clicked(lambda x: self.load_tiles())
        self.btn_zoom_in.on_clicked(lambda x: self.zoom_in())
        self.btn_zoom_out.on_clicked(lambda x: self.zoom_out())
        
        # Add instructions
        instructions = [
            "🔧 NAVIGATION:",
            "Use toolbar OR + - buttons",
            "",
            "🎯 DRAW ONE WATER ZONE:",
            "Left-click: Add vertex", 
            "Right-click: Close zone",
            "",
            "📋 CONTROLS:",
            "'s': Save zone",
            "'c': Clear current",
            "'u': Undo point",
            "'r': Refresh tiles",
            "'q': Quit",
            "",
            "🗺️ NAVIGATION TIPS:",
            "1. Use + - buttons to zoom",
            "2. Use toolbar pan tool to move",
            "3. Tiles reload automatically"
        ]
        
        text = '\n'.join(instructions)
        self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightblue', 
                            alpha=0.95, edgecolor='navy', linewidth=2),
                    zorder=20)
    
    def on_view_changed(self, ax):
        """Handle view changes from toolbar zoom/pan"""
        # Get current view bounds
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        
        # Initialize if first time
        if self.last_xlim is None or self.last_ylim is None:
            self.last_xlim = current_xlim
            self.last_ylim = current_ylim
            return
        
        # Check for significant changes
        xlim_change = abs(current_xlim[0] - self.last_xlim[0]) + abs(current_xlim[1] - self.last_xlim[1])
        ylim_change = abs(current_ylim[0] - self.last_ylim[0]) + abs(current_ylim[1] - self.last_ylim[1])
        
        if xlim_change > 0.001 or ylim_change > 0.001:  # Significant change
            print("🗺️ View changed, reloading street map tiles...")
            try:
                self.load_tiles()
            except Exception as e:
                print(f"Error reloading tiles: {e}")
        
        self.last_xlim = current_xlim
        self.last_ylim = current_ylim
    
    def on_mouse_release(self, event):
        """Handle mouse release events (for detecting end of pan operations)"""
        # Check if we were in pan mode and view has changed
        toolbar = self.fig.canvas.toolbar
        if toolbar and toolbar.mode == 'pan/zoom':
            current_xlim = self.ax.get_xlim()
            current_ylim = self.ax.get_ylim()
            
            # Check if view actually changed
            if (self.last_xlim and self.last_ylim and
                (abs(current_xlim[0] - self.last_xlim[0]) > 0.001 or
                 abs(current_xlim[1] - self.last_xlim[1]) > 0.001 or
                 abs(current_ylim[0] - self.last_ylim[0]) > 0.001 or
                 abs(current_ylim[1] - self.last_ylim[1]) > 0.001)):
                
                print("🗺️ Pan completed, reloading tiles...")
                self.load_tiles()
                self.last_xlim = current_xlim
                self.last_ylim = current_ylim
    
    def zoom_in(self):
        """Zoom in by reducing the view bounds"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Zoom by 50%
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * 0.5
        y_range = (ylim[1] - ylim[0]) * 0.5
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        print("🔍 Zooming in...")
        self.load_tiles()
        self.fig.canvas.draw()
    
    def zoom_out(self):
        """Zoom out by expanding the view bounds"""
        xlim = self.ax.get_xlim()
        ylim = self.ax.get_ylim()
        
        # Zoom out by 50%
        x_center = (xlim[0] + xlim[1]) / 2
        y_center = (ylim[0] + ylim[1]) / 2
        x_range = (xlim[1] - xlim[0]) * 1.5
        y_range = (ylim[1] - ylim[0]) * 1.5
        
        self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
        self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        print("🔍 Zooming out...")
        self.load_tiles()
        self.fig.canvas.draw()
    
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        # Check if toolbar is in navigation mode - don't draw while navigating
        toolbar = self.fig.canvas.toolbar
        if toolbar and toolbar.mode != '':
            return  # Don't draw while zooming or panning
        
        # Get coordinates
        lon, lat = event.xdata, event.ydata
        
        if event.button == 1:  # Left click - add point
            self.current_polygon.append([lon, lat])
            
            # Plot the point
            point, = self.ax.plot(lon, lat, 'o', color='lime', markersize=15, 
                                 markeredgecolor='darkgreen', markeredgewidth=3,
                                 zorder=25)
            self.temp_points.append(point)
            
            # Draw line from previous point
            if len(self.current_polygon) > 1:
                prev_point = self.current_polygon[-2]
                line, = self.ax.plot([prev_point[0], lon], [prev_point[1], lat], 
                                    'g-', linewidth=5, alpha=0.8, zorder=22)
                self.temp_lines.append(line)
            
            print(f"✅ Point {len(self.current_polygon)}: ({lon:.5f}, {lat:.5f})")
            
        elif event.button == 3:  # Right click - close polygon
            if len(self.current_polygon) >= 3:
                # Only allow one polygon
                if self.saved_polygons:
                    print("❌ Only one inclusion zone allowed. Clear existing zone first.")
                    return
                
                # Close the polygon
                self.current_polygon.append(self.current_polygon[0])
                
                # Draw closing line
                last_point = self.current_polygon[-2]
                first_point = self.current_polygon[0]
                line, = self.ax.plot([last_point[0], first_point[0]], 
                                    [last_point[1], first_point[1]], 
                                    'g-', linewidth=5, alpha=0.8, zorder=22)
                self.temp_lines.append(line)
                
                # Create polygon patch
                polygon = Polygon(self.current_polygon, 
                                closed=True, alpha=0.3, 
                                facecolor='cyan', edgecolor='blue',
                                linewidth=4, zorder=15)
                self.ax.add_patch(polygon)
                
                # Save the polygon
                self.saved_polygons.append({
                    'type': 'inclusion',
                    'coordinates': self.current_polygon.copy()
                })
                
                print(f"🔵 Water inclusion zone created! {len(self.current_polygon)-1} vertices")
                print("💾 Press 's' to save")
                
                # Clear current
                self.current_polygon = []
                self.clear_temp_drawings()
            else:
                print("❌ Need at least 3 points")
        
        self.fig.canvas.draw()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 's':
            self.save_polygons()
        elif event.key == 'c':
            self.clear_current()
        elif event.key == 'u':
            self.undo_point()
        elif event.key == 'r':
            self.load_tiles()
        elif event.key == 'q':
            plt.close()
    
    def clear_current(self):
        """Clear current polygon"""
        self.current_polygon = []
        self.clear_temp_drawings()
        print("🧹 Cleared current polygon")
        self.fig.canvas.draw()
    
    def undo_point(self):
        """Undo last point"""
        if self.current_polygon:
            self.current_polygon.pop()
            if self.temp_lines:
                line = self.temp_lines.pop()
                line.remove()
            if self.temp_points:
                point = self.temp_points.pop()
                point.remove()
            print(f"↩️ Undid point. {len(self.current_polygon)} remaining")
            self.fig.canvas.draw()
    
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
    
    def redraw_all(self):
        """Redraw all permanent elements"""
        # Redraw saved polygons
        for i, poly_data in enumerate(self.saved_polygons):
            polygon = Polygon(poly_data['coordinates'], 
                            closed=True, alpha=0.3, 
                            facecolor='cyan', edgecolor='blue',
                            linewidth=4, zorder=15)
            self.ax.add_patch(polygon)
        
        # Ensure reefs are on top
        self.reef_scatter.set_zorder(20)
        for text in self.reef_texts:
            text.set_zorder(21)
    
    def save_polygons(self):
        """Save polygons to file"""
        if not self.saved_polygons:
            print("❌ No polygons to save")
            return
        
        os.makedirs('data', exist_ok=True)
        filename = 'data/inclusion_zones.json'
        
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
        
        with open(filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"\n💾 SAVED {len(self.saved_polygons)} zones to {filename}")
        print("🎉 Ready for Streamlit!")
    
    def run(self):
        """Run the application"""
        # Load existing zones
        try:
            with open('data/inclusion_zones.json', 'r') as f:
                data = json.load(f)
                if 'zones' in data:
                    self.saved_polygons = data['zones']
                    print(f"✅ Loaded {len(self.saved_polygons)} existing zones")
        except:
            pass
        
        # Remove tight_layout to avoid warnings
        plt.show()

def main():
    print("="*60)
    print("🗺️ STREET MAP DRAWING TOOL WITH NAVIGATION TOOLBAR")
    print("="*60)
    print("\n🔧 NAVIGATION TOOLBAR will appear at the bottom")
    print("   ⚡ Use zoom/pan tools in toolbar - tiles auto-reload")
    print("\n🖱️ DRAWING:")
    print("   Left-click to add points, right-click to close polygon")
    print("   Press 's' to save zone when done")
    print("\nLoading OpenStreetMap tiles...")
    print("="*60)
    
    app = InteractiveMapDrawer()
    app.run()

if __name__ == "__main__":
    main()