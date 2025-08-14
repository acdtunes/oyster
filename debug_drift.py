#!/usr/bin/env python3
"""
Debug why dispersal appears circular when currents should create drift
"""

import pandas as pd
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt

# Load reef data
reef_data = pd.read_csv('output/st_marys/reef_metrics.csv').iloc[:28]

# Calculate drift for all reefs
with nc.Dataset('data/109516.nc', 'r') as ds:
    u_raw = ds.variables['u_surface'][:]
    v_raw = ds.variables['v_surface'][:]
    lon = ds.variables['longitude'][:]
    lat = ds.variables['latitude'][:]
    
    # Average over time
    u_mean = np.mean(u_raw, axis=0)
    v_mean = np.mean(v_raw, axis=0)
    
    source_lons = []
    source_lats = []
    drift_lons = []
    drift_lats = []
    
    for _, reef in reef_data.iterrows():
        source_lon = reef['Longitude']
        source_lat = reef['Latitude']
        
        lon_idx = np.argmin(np.abs(lon - source_lon))
        lat_idx = np.argmin(np.abs(lat - source_lat))
        
        u_at_source = u_mean[lat_idx, lon_idx]
        v_at_source = v_mean[lat_idx, lon_idx]
        
        if np.isnan(u_at_source):
            u_at_source = 0
        if np.isnan(v_at_source):
            v_at_source = 0
        
        # Calculate drift
        meters_per_degree_lon = 111000 * np.cos(np.radians(source_lat))
        meters_per_degree_lat = 111000
        
        u_deg_per_day = (u_at_source * 86400) / meters_per_degree_lon
        v_deg_per_day = (v_at_source * 86400) / meters_per_degree_lat
        
        drift_lon = source_lon + (u_deg_per_day * 21)
        drift_lat = source_lat + (v_deg_per_day * 21)
        
        source_lons.append(source_lon)
        source_lats.append(source_lat)
        drift_lons.append(drift_lon)
        drift_lats.append(drift_lat)

# Plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot source reefs
ax.scatter(source_lons, source_lats, c='green', s=100, label='Source Reefs', zorder=3)

# Plot drift endpoints
ax.scatter(drift_lons, drift_lats, c='red', s=100, label='Drift Endpoints (Day 21)', zorder=3)

# Draw drift vectors
for i in range(len(source_lons)):
    ax.arrow(source_lons[i], source_lats[i], 
             drift_lons[i] - source_lons[i], 
             drift_lats[i] - source_lats[i],
             head_width=0.01, head_length=0.01, 
             fc='blue', ec='blue', alpha=0.5)

# Show map bounds
lon_center = np.mean(source_lons)
lat_center = np.mean(source_lats)
lon_buffer = 0.05
lat_buffer = 0.08

# Display area boundary
rect = plt.Rectangle((lon_center - lon_buffer, lat_center - lat_buffer),
                     2*lon_buffer, 2*lat_buffer,
                     fill=False, edgecolor='black', linewidth=2,
                     linestyle='--', label='Display Area')
ax.add_patch(rect)

ax.set_xlabel('Longitude')
ax.set_ylabel('Latitude')
ax.set_title('Larval Drift Problem: Kernels Drift Outside Display Area!')
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

# Add annotations
ax.text(lon_center, lat_center + lat_buffer + 0.02, 
        'This is the area\nshown in Streamlit', 
        ha='center', fontsize=10, color='red')

mean_drift_lon = np.mean(drift_lons)
mean_drift_lat = np.mean(drift_lats)
ax.text(mean_drift_lon, mean_drift_lat - 0.05,
        'Dispersal kernels\nare centered HERE\n(~40km away!)',
        ha='center', fontsize=10, color='red')

plt.tight_layout()
plt.savefig('debug_drift_problem.png', dpi=150)
plt.show()

print(f"Source reef center: ({lon_center:.3f}, {lat_center:.3f})")
print(f"Mean drift endpoint: ({mean_drift_lon:.3f}, {mean_drift_lat:.3f})")
print(f"Distance between centers: {np.sqrt((mean_drift_lon-lon_center)**2 + (mean_drift_lat-lat_center)**2)*111:.1f} km")
print(f"\nPROBLEM: The dispersal kernels are centered at the drift endpoints,")
print(f"which are OUTSIDE the displayed map area!")
print(f"That's why you only see the edge of the kernels as circles.")