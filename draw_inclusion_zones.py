#!/usr/bin/env python3
"""
Desktop application for drawing inclusion zones on a map
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

class PolygonDrawer:
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
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        # Set up the map
        self.setup_map()
        
        # Connect event handlers
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Drawing state
        self.drawing = False
        self.temp_lines = []
        
    def setup_map(self):
        """Set up the map with proper coordinates"""
        # Set limits
        self.ax.set_xlim(self.lon_min, self.lon_max)
        self.ax.set_ylim(self.lat_min, self.lat_max)
        
        # Labels
        self.ax.set_xlabel('Longitude', fontsize=12)
        self.ax.set_ylabel('Latitude', fontsize=12)
        self.ax.set_title('Draw Inclusion Zones - Click to add points, Right-click to close polygon', fontsize=14)
        
        # Grid
        self.ax.grid(True, alpha=0.3)
        
        # Plot reef locations as reference
        reef_subset = self.reef_data.iloc[:27]  # First 27 reefs
        self.ax.scatter(reef_subset['Longitude'], reef_subset['Latitude'], 
                       c='red', s=50, marker='o', label='Reef Sites', zorder=5)
        
        # Add reef labels
        for _, reef in reef_subset.iterrows():
            self.ax.annotate(reef['SourceReef'], 
                           (reef['Longitude'], reef['Latitude']),
                           fontsize=6, ha='center', va='bottom')
        
        # Aspect ratio to match lat/lon scaling
        self.ax.set_aspect(1/np.cos(np.radians(self.lat_min + self.lat_max)/2))
        
        # Legend
        self.ax.legend(loc='upper right')
        
        # Instructions
        self.add_instructions()
        
    def add_instructions(self):
        """Add instructions to the plot"""
        instructions = [
            "INSTRUCTIONS:",
            "• Left-click: Add vertex",
            "• Right-click: Close polygon",
            "• 'c': Clear current",
            "• 's': Save all polygons",
            "• 'u': Undo last point",
            "• 'd': Delete last polygon",
            "• 'q': Quit"
        ]
        
        text = '\n'.join(instructions)
        self.ax.text(0.02, 0.98, text, transform=self.ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
    def on_click(self, event):
        """Handle mouse clicks"""
        if event.inaxes != self.ax:
            return
        
        # Get coordinates
        lon, lat = event.xdata, event.ydata
        
        if event.button == 1:  # Left click - add point
            self.current_polygon.append([lon, lat])
            
            # Plot the point
            self.ax.plot(lon, lat, 'bo', markersize=8)
            
            # Draw line from previous point
            if len(self.current_polygon) > 1:
                prev_point = self.current_polygon[-2]
                line, = self.ax.plot([prev_point[0], lon], [prev_point[1], lat], 'b-', linewidth=2)
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
                self.ax.plot([last_point[0], first_point[0]], 
                           [last_point[1], first_point[1]], 'b-', linewidth=2)
                
                # Create polygon patch
                polygon = Polygon(self.current_polygon, 
                                closed=True, 
                                alpha=0.3, 
                                facecolor='cyan',
                                edgecolor='blue',
                                linewidth=2)
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
                print(f"Removed last point. {len(self.current_polygon)} points remaining")
                plt.draw()
                
        elif event.key == 'd':  # Delete last saved polygon
            if self.saved_polygons:
                self.saved_polygons.pop()
                # Redraw everything
                self.ax.clear()
                self.setup_map()
                self.redraw_polygons()
                print(f"Deleted last polygon. {len(self.saved_polygons)} polygons remaining")
                plt.draw()
                
        elif event.key == 's':  # Save polygons to file
            self.save_polygons()
            
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
        
        # Clear points
        # This is tricky - would need to track point artists
        
    def redraw_polygons(self):
        """Redraw all saved polygons"""
        for poly_data in self.saved_polygons:
            polygon = Polygon(poly_data['coordinates'], 
                            closed=True, 
                            alpha=0.3, 
                            facecolor='cyan',
                            edgecolor='blue',
                            linewidth=2)
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
        
        print(f"Saved {len(self.saved_polygons)} inclusion zones to {filename}")
        
        # Also save a backup with timestamp
        backup_filename = f'data/inclusion_zones_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(backup_filename, 'w') as f:
            json.dump(save_data, f, indent=2)
        print(f"Backup saved to {backup_filename}")
    
    def run(self):
        """Run the application"""
        # Load existing zones if available
        try:
            with open('data/inclusion_zones.json', 'r') as f:
                data = json.load(f)
                if 'zones' in data:
                    self.saved_polygons = data['zones']
                    self.redraw_polygons()
                    print(f"Loaded {len(self.saved_polygons)} existing zones")
        except FileNotFoundError:
            print("No existing zones found")
        except Exception as e:
            print(f"Error loading zones: {e}")
        
        plt.show()

def main():
    print("="*60)
    print("INCLUSION ZONE DRAWING TOOL")
    print("="*60)
    print("\nThis tool allows you to draw polygons on the map.")
    print("These polygons define INCLUSION zones - areas where")
    print("settlement calculations WILL be performed (water areas).")
    print("\nThe coordinates are in latitude/longitude.")
    print("\nStarting application...")
    print("="*60)
    
    app = PolygonDrawer()
    app.run()
    
    print("\nApplication closed.")

if __name__ == "__main__":
    main()