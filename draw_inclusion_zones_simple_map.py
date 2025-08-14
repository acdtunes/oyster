#!/usr/bin/env python3
"""
Desktop application for drawing inclusion zones with simple background
Uses bathymetry/water boundaries to show where water is
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
import json
import pandas as pd
from datetime import datetime

class PolygonDrawerSimple:
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
        
        # Create background that hints at water areas
        self.create_water_background()
        
        # Set up the map
        self.setup_map()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Drawing state
        self.drawing = False
        self.temp_lines = []
        self.temp_points = []
        
    def create_water_background(self):
        """Create a background that suggests water areas based on known bathymetry"""
        # Create a bathymetry-like background
        lon_range = np.linspace(self.lon_min, self.lon_max, 200)
        lat_range = np.linspace(self.lat_min, self.lat_max, 200)
        lon_grid, lat_grid = np.meshgrid(lon_range, lat_range)
        
        # Create a simple "water depth" field that follows the river shape
        # St. Mary's River runs roughly NE-SW
        
        # Main river channel (deeper water)
        main_channel_depth = np.zeros_like(lon_grid)
        
        # Define main channel path (rough approximation of St. Mary's River)
        river_points = [
            [-76.49, 38.13],   # Mouth
            [-76.475, 38.145],
            [-76.46, 38.16],
            [-76.45, 38.175],
            [-76.44, 38.19],
            [-76.425, 38.205],
            [-76.41, 38.22]    # Upstream
        ]
        
        # Create river channel by adding "depth" around the path
        for i in range(len(river_points)-1):
            start_lon, start_lat = river_points[i]
            end_lon, end_lat = river_points[i+1]
            
            # Create line segment
            t = np.linspace(0, 1, 50)
            seg_lon = start_lon + t * (end_lon - start_lon)
            seg_lat = start_lat + t * (end_lat - start_lat)
            
            # Add "depth" around this segment
            for lon_pt, lat_pt in zip(seg_lon, seg_lat):
                dist_from_center = np.sqrt((lon_grid - lon_pt)**2 * 10000 + 
                                         (lat_grid - lat_pt)**2 * 10000)
                # Create channel width that varies
                channel_width = 0.015 + 0.005 * np.random.normal(0, 0.1, dist_from_center.shape)
                depth_contribution = np.exp(-dist_from_center**2 / (2 * channel_width**2))
                main_channel_depth += depth_contribution
        
        # Add some tributaries
        tributaries = [
            [[-76.465, 38.14], [-76.455, 38.135]],  # Small tributary
            [[-76.44, 38.175], [-76.435, 38.17]],   # Another tributary
            [[-76.42, 38.2], [-76.415, 38.195]]     # Upstream tributary
        ]
        
        for trib in tributaries:
            start_lon, start_lat = trib[0]
            end_lon, end_lat = trib[1]
            
            t = np.linspace(0, 1, 20)
            trib_lon = start_lon + t * (end_lon - start_lon)
            trib_lat = start_lat + t * (end_lat - start_lat)
            
            for lon_pt, lat_pt in zip(trib_lon, trib_lat):
                dist_from_center = np.sqrt((lon_grid - lon_pt)**2 * 10000 + 
                                         (lat_grid - lat_pt)**2 * 10000)
                depth_contribution = np.exp(-dist_from_center**2 / (2 * 0.008**2)) * 0.5
                main_channel_depth += depth_contribution
        
        # Normalize and create color map
        main_channel_depth = np.clip(main_channel_depth, 0, 1)
        
        # Create water-like colormap (blues for water, browns for land)
        water_mask = main_channel_depth > 0.1
        
        # Display as background
        colors = np.zeros((*main_channel_depth.shape, 3))
        # Land areas (brown/tan)
        colors[:, :, 0] = 0.8  # Red
        colors[:, :, 1] = 0.7  # Green  
        colors[:, :, 2] = 0.5  # Blue
        
        # Water areas (blue)
        colors[water_mask, 0] = 0.2 * (1 - main_channel_depth[water_mask])  # Less red in water
        colors[water_mask, 1] = 0.4 * (1 - main_channel_depth[water_mask])  # Less green in water
        colors[water_mask, 2] = 0.6 + 0.3 * main_channel_depth[water_mask]   # More blue in deeper water
        
        self.ax.imshow(colors, extent=[self.lon_min, self.lon_max, self.lat_min, self.lat_max], 
                      aspect='auto', alpha=0.7, zorder=0, origin='lower')
        
        print("Background water map created!")
        
    def setup_map(self):
        """Set up the map with proper coordinates"""
        # Set limits
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        
        # Labels
        self.ax.set_xlabel('Longitude', fontsize=14, fontweight='bold')
        self.ax.set_ylabel('Latitude', fontsize=14, fontweight='bold')
        self.ax.set_title('Draw Water Inclusion Zones on St. Mary\'s River\n(Blue areas suggest water, brown areas suggest land)', 
                         fontsize=16, fontweight='bold')
        
        # Grid
        self.ax.grid(True, alpha=0.5, color='white', linewidth=1)
        
        # Plot reef locations as reference
        reef_subset = self.reef_data.iloc[:27]  # First 27 reefs
        self.ax.scatter(reef_subset['Longitude'], reef_subset['Latitude'], 
                       c='red', s=150, marker='*', label='Reef Sites', 
                       zorder=10, edgecolors='darkred', linewidth=2)
        
        # Add reef labels with better visibility
        for _, reef in reef_subset.iterrows():
            self.ax.annotate(reef['SourceReef'], 
                           (reef['Longitude'], reef['Latitude']),
                           fontsize=9, ha='center', va='bottom',
                           color='darkred', fontweight='bold',
                           bbox=dict(boxstyle='round,pad=0.3', 
                                   facecolor='yellow', alpha=0.8))
        
        # Aspect ratio to match lat/lon scaling
        self.ax.set_aspect(1/np.cos(np.radians((self.lat_min + self.lat_max)/2)))
        
        # Legend
        self.ax.legend(loc='upper right', fontsize=12, framealpha=0.9)
        
        # Instructions
        self.add_instructions()
        
    def add_instructions(self):
        """Add instructions to the plot"""
        instructions = [
            "🌊 DRAW WATER AREAS:",
            "",
            "• Left-click: Add vertex",
            "• Right-click: Close polygon", 
            "• 'c': Clear current drawing",
            "• 's': SAVE all zones",
            "• 'u': Undo last point",
            "• 'd': Delete last polygon",
            "• 'l': Load existing zones",
            "• 'q': Quit",
            "",
            "🎯 Draw polygons around",
            "   BLUE water areas only!",
            "",
            "⭐ Red stars = Reef sites"
        ]
        
        text = '\n'.join(instructions)
        self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                    fontsize=11, verticalalignment='top', fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightcyan', 
                            alpha=0.95, edgecolor='navy', linewidth=3))
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        # Get coordinates
        lon, lat = event.xdata, event.ydata
        
        if event.button == 1:  # Left click - add point
            self.current_polygon.append([lon, lat])
            
            # Plot the point
            point, = self.ax.plot(lon, lat, 'o', color='lime', markersize=12, 
                                 markeredgecolor='darkgreen', markeredgewidth=3,
                                 zorder=15)
            self.temp_points.append(point)
            
            # Draw line from previous point
            if len(self.current_polygon) > 1:
                prev_point = self.current_polygon[-2]
                line, = self.ax.plot([prev_point[0], lon], [prev_point[1], lat], 
                                    'g-', linewidth=4, alpha=0.8, zorder=12)
                self.temp_lines.append(line)
            
            # Update status
            print(f"✅ Added point {len(self.current_polygon)}: ({lon:.5f}, {lat:.5f})")
            
        elif event.button == 3:  # Right click - close polygon
            if len(self.current_polygon) >= 3:
                # Close the polygon
                self.current_polygon.append(self.current_polygon[0])
                
                # Draw closing line
                last_point = self.current_polygon[-2]
                first_point = self.current_polygon[0]
                line, = self.ax.plot([last_point[0], first_point[0]], 
                                    [last_point[1], first_point[1]], 
                                    'g-', linewidth=4, alpha=0.8, zorder=12)
                self.temp_lines.append(line)
                
                # Create polygon patch
                polygon = Polygon(self.current_polygon, 
                                closed=True, 
                                alpha=0.4, 
                                facecolor='cyan',
                                edgecolor='blue',
                                linewidth=4,
                                zorder=5)
                self.ax.add_patch(polygon)
                
                # Save the polygon
                self.saved_polygons.append({
                    'type': 'inclusion',
                    'coordinates': self.current_polygon.copy()
                })
                
                print(f"🔵 Closed polygon with {len(self.current_polygon)-1} vertices")
                print(f"📊 Total polygons: {len(self.saved_polygons)}")
                
                # Clear current polygon
                self.current_polygon = []
                self.clear_temp_drawings()
            else:
                print("❌ Need at least 3 points to close polygon")
        
        plt.draw()
    
    def on_key(self, event):
        """Handle keyboard events"""
        if event.key == 'c':  # Clear current polygon
            self.current_polygon = []
            self.clear_temp_drawings()
            print("🧹 Cleared current polygon")
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
                print(f"↩️ Removed last point. {len(self.current_polygon)} points remaining")
                plt.draw()
                
        elif event.key == 'd':  # Delete last saved polygon
            if self.saved_polygons:
                self.saved_polygons.pop()
                # Redraw everything
                self.ax.clear()
                self.create_water_background()
                self.setup_map()
                self.redraw_polygons()
                print(f"🗑️ Deleted last polygon. {len(self.saved_polygons)} polygons remaining")
                plt.draw()
                
        elif event.key == 's':  # Save polygons to file
            self.save_polygons()
            
        elif event.key == 'l':  # Load existing zones
            self.load_existing_zones()
            
        elif event.key == 'q':  # Quit
            print("👋 Goodbye!")
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
        for i, poly_data in enumerate(self.saved_polygons):
            polygon = Polygon(poly_data['coordinates'], 
                            closed=True, 
                            alpha=0.4, 
                            facecolor='cyan',
                            edgecolor='blue',
                            linewidth=4,
                            zorder=5)
            self.ax.add_patch(polygon)
            
            # Add zone number
            coords = np.array(poly_data['coordinates'])
            center_lon = coords[:, 0].mean()
            center_lat = coords[:, 1].mean()
            self.ax.text(center_lon, center_lat, f'Z{i+1}', 
                        ha='center', va='center', fontsize=12, fontweight='bold',
                        color='darkblue', 
                        bbox=dict(boxstyle='circle', facecolor='white', alpha=0.8))
    
    def save_polygons(self):
        """Save polygons to JSON file"""
        if not self.saved_polygons:
            print("❌ No polygons to save")
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
        
        print(f"\n💾 SAVED {len(self.saved_polygons)} inclusion zones to {filename}")
        
        # Also save a backup with timestamp
        backup_filename = f'data/inclusion_zones_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"📁 Backup saved to {backup_filename}")
        print("🎉 Ready to use in Streamlit app!")
    
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
        
        plt.tight_layout()
        plt.show()

def main():
    print("="*60)
    print("🦪 WATER INCLUSION ZONE DRAWING TOOL")
    print("="*60)
    print("\n🎯 Draw polygons around WATER AREAS (blue regions)")
    print("   where oyster larvae can settle.")
    print("\n⭐ Red stars show existing reef locations.")
    print("🔵 Blue areas suggest water, brown areas suggest land.")
    print("\n🖱️  Left-click to add points, right-click to close polygon")
    print("⌨️  Press 's' to save when done!")
    print("\nStarting application...")
    print("="*60)
    
    app = PolygonDrawerSimple()
    app.run()
    
    print("\n👋 Application closed.")

if __name__ == "__main__":
    main()