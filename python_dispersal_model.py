#!/usr/bin/env python3
"""
Python implementation of the current-based larval dispersal model
Matches the R dispersal_modeling.R logic for consistency
"""

import numpy as np
import netCDF4 as nc
from scipy.spatial.distance import cdist
import pandas as pd

def calculate_advection_diffusion_settlement(reef_data, nc_file='data/109516.nc', 
                                            pelagic_duration=21, mortality_rate=0.1,
                                            diffusion_coeff=100, settlement_day=14):
    """
    Calculate larval settlement probability field using advection-diffusion model
    with ocean currents, matching the R model implementation.
    
    Args:
        reef_data: DataFrame with reef locations (Longitude, Latitude, Density)
        nc_file: Path to NetCDF file with current data
        pelagic_duration: Days larvae spend in water column (21)
        mortality_rate: Daily mortality rate (0.1 = 10%)
        diffusion_coeff: Horizontal diffusion coefficient (m²/s)
        settlement_day: Day when larvae become competent to settle (14)
    
    Returns:
        lon_grid, lat_grid, settlement_prob: Coordinate grids and probability field
    """
    
    # Load current data from NetCDF
    with nc.Dataset(nc_file, 'r') as dataset:
        nc_lon = dataset.variables['longitude'][:]
        nc_lat = dataset.variables['latitude'][:]
        
        # Get current velocities (m/s)
        # Check the actual shape to handle correctly
        u_surface_raw = dataset.variables['u_surface'][:]
        v_surface_raw = dataset.variables['v_surface'][:]
        
        # The data is [time, lat, lon] ordering
        # If it's 3D, average over time (first dimension)
        if len(u_surface_raw.shape) == 3:
            # Assuming [time, lat, lon] ordering
            u_mean = np.mean(u_surface_raw, axis=0)  # Average over time
            v_mean = np.mean(v_surface_raw, axis=0)
        else:
            u_mean = u_surface_raw
            v_mean = v_surface_raw
        
        # Get water mask
        water_mask = dataset.variables['mask_land_sea'][:]  # Shape: [lat, lon]
    
    # Create output grid that includes BOTH source reefs AND drift endpoints
    # First, calculate where all larvae will drift to
    all_drift_lons = []
    all_drift_lats = []
    
    for _, reef in reef_data.iterrows():
        source_lon = reef['Longitude']
        source_lat = reef['Latitude']
        
        # Find nearest grid point for current extraction
        lon_idx = np.argmin(np.abs(nc_lon - source_lon))
        lat_idx = np.argmin(np.abs(nc_lat - source_lat))
        
        # Get currents
        u_at_source = u_mean[lat_idx, lon_idx]
        v_at_source = v_mean[lat_idx, lon_idx]
        
        if np.isnan(u_at_source):
            u_at_source = 0
        if np.isnan(v_at_source):
            v_at_source = 0
        
        # Calculate drift endpoint
        meters_per_degree_lon = 111000 * np.cos(np.radians(source_lat))
        meters_per_degree_lat = 111000
        
        u_deg_per_day = (u_at_source * 86400) / meters_per_degree_lon
        v_deg_per_day = (v_at_source * 86400) / meters_per_degree_lat
        
        drift_lon = source_lon + (u_deg_per_day * pelagic_duration)
        drift_lat = source_lat + (v_deg_per_day * pelagic_duration)
        
        all_drift_lons.append(drift_lon)
        all_drift_lats.append(drift_lat)
    
    # Now set grid bounds to include BOTH sources and drift endpoints
    all_lons = list(reef_data['Longitude']) + all_drift_lons
    all_lats = list(reef_data['Latitude']) + all_drift_lats
    
    lon_min = min(all_lons) - 0.05  # Add buffer
    lon_max = max(all_lons) + 0.05
    lat_min = min(all_lats) - 0.15  # EXPANDED southward to cover more of Chesapeake Bay
    lat_max = max(all_lats) + 0.05
    
    print(f"Grid bounds: lon [{lon_min:.3f}, {lon_max:.3f}], lat [{lat_min:.3f}, {lat_max:.3f}]")
    
    # Moderate resolution output grid for better performance
    n_lon = 200  # Balanced for performance
    n_lat = 300  # Balanced for performance
    lon_grid = np.linspace(lon_min, lon_max, n_lon)
    lat_grid = np.linspace(lat_min, lat_max, n_lat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Initialize settlement probability field
    settlement_prob = np.zeros((n_lat, n_lon))
    
    # For each source reef, calculate its contribution
    for _, reef in reef_data.iterrows():
        source_lon = reef['Longitude']
        source_lat = reef['Latitude']
        source_density = reef['Density'] if 'Density' in reef else reef.get('AvgDensity', 100)
        
        # Find nearest grid point in NetCDF data for current extraction
        lon_idx = np.argmin(np.abs(nc_lon - source_lon))
        lat_idx = np.argmin(np.abs(nc_lat - source_lat))
        
        # Extract currents at source location
        # Access as [lat_idx, lon_idx] since data is [lat, lon]
        u_at_source = u_mean[lat_idx, lon_idx]
        v_at_source = v_mean[lat_idx, lon_idx]
        
        # Handle missing data
        if np.isnan(u_at_source):
            u_at_source = 0
        if np.isnan(v_at_source):
            v_at_source = 0
        
        # Calculate larval drift due to currents
        # Convert m/s to degrees per day
        meters_per_degree_lon = 111000 * np.cos(np.radians(source_lat))
        meters_per_degree_lat = 111000
        
        # Use actual currents without scaling - model must reflect reality
        u_deg_per_day = (u_at_source * 86400) / meters_per_degree_lon
        v_deg_per_day = (v_at_source * 86400) / meters_per_degree_lat
        
        # Where larvae end up after drifting with current for pelagic_duration days
        drift_lon = source_lon + (u_deg_per_day * pelagic_duration)
        drift_lat = source_lat + (v_deg_per_day * pelagic_duration)
        
        # Calculate diffusive spread
        # In reality, larvae are released continuously and experience variable currents
        # This creates a plume from source to drift endpoint, not just a point at the end
        
        # Effective diffusion includes both turbulent mixing and variability in currents
        # Use enhanced diffusion to represent tidal variability and continuous release
        effective_diffusion = diffusion_coeff * 10  # Account for tidal dispersion
        
        diffusion_deg2_per_day = effective_diffusion / (meters_per_degree_lon * meters_per_degree_lat)
        
        # Total variance after pelagic_duration days
        variance = 2 * diffusion_deg2_per_day * pelagic_duration
        
        # Standard deviation in degrees
        sigma = np.sqrt(variance)
        
        # Ensure reasonable minimum spread
        sigma = max(sigma, 0.05)  # Minimum ~5 km spread
        
        # Calculate survival probability after pelagic_duration days
        survival_prob = np.exp(-mortality_rate * pelagic_duration)
        
        # For each grid point, calculate probability of larvae from this source
        for i in range(n_lat):
            for j in range(n_lon):
                target_lon = lon_grid[j]
                target_lat = lat_grid[i]
                
                # Create a dispersal plume from source to drift endpoint
                # This represents continuous larval release over spawning period
                
                # Method 1: Probability at drift endpoint (main concentration)
                lon_dist_drift = (target_lon - drift_lon) * np.cos(np.radians(target_lat)) * 111
                lat_dist_drift = (target_lat - drift_lat) * 111
                dist_from_drift_km = np.sqrt(lon_dist_drift**2 + lat_dist_drift**2)
                
                # Method 2: Probability along drift path (dispersal corridor)
                # Calculate distance from the line between source and drift endpoint
                # This creates an elongated plume along the current path
                
                # Vector from source to drift
                drift_vec_lon = drift_lon - source_lon
                drift_vec_lat = drift_lat - source_lat
                drift_distance = np.sqrt((drift_vec_lon * 111 * np.cos(np.radians(source_lat)))**2 + 
                                       (drift_vec_lat * 111)**2)
                
                if drift_distance > 0.1:  # If there's significant drift
                    # Project point onto drift line
                    t = max(0, min(1, ((target_lon - source_lon) * drift_vec_lon + 
                                      (target_lat - source_lat) * drift_vec_lat) / 
                                     (drift_vec_lon**2 + drift_vec_lat**2)))
                    
                    # Closest point on drift line
                    closest_lon = source_lon + t * drift_vec_lon
                    closest_lat = source_lat + t * drift_vec_lat
                    
                    # Distance from drift line
                    lon_dist_line = (target_lon - closest_lon) * np.cos(np.radians(target_lat)) * 111
                    lat_dist_line = (target_lat - closest_lat) * 111
                    dist_from_line_km = np.sqrt(lon_dist_line**2 + lat_dist_line**2)
                    
                    # Combine: higher probability near drift line AND drift endpoint
                    # This creates a plume that extends from source to destination
                    prob_line = np.exp(-dist_from_line_km**2 / (2 * (sigma * 111)**2))
                    prob_endpoint = np.exp(-dist_from_drift_km**2 / (2 * (sigma * 111)**2))
                    
                    # Weight: more larvae accumulate toward the drift endpoint
                    weight_along_path = 0.3  # 30% distributed along path
                    weight_at_endpoint = 0.7  # 70% concentrated at endpoint
                    
                    prob = weight_along_path * prob_line + weight_at_endpoint * prob_endpoint
                else:
                    # No significant drift - just use radial spread from source
                    lon_dist = (target_lon - source_lon) * np.cos(np.radians(target_lat)) * 111
                    lat_dist = (target_lat - source_lat) * 111
                    dist_km = np.sqrt(lon_dist**2 + lat_dist**2)
                    prob = np.exp(-dist_km**2 / (2 * (sigma * 111)**2))
                
                # Scale by source strength and survival
                prob *= (source_density / 100) * survival_prob
                
                # Add to total settlement probability
                settlement_prob[i, j] += prob
    
    # Normalize to [0, 1]
    if settlement_prob.max() > 0:
        settlement_prob = settlement_prob / settlement_prob.max()
    
    return lon_grid, lat_grid, settlement_prob


def get_current_at_location(lon, lat, u_field, v_field, nc_lon, nc_lat):
    """
    Extract current velocity at a specific location
    
    Args:
        lon, lat: Location coordinates
        u_field, v_field: Current velocity fields [lat, lon]
        nc_lon, nc_lat: NetCDF coordinate arrays
    
    Returns:
        u, v: Current velocities at location (m/s)
    """
    # Find nearest grid point
    lon_idx = np.argmin(np.abs(nc_lon - lon))
    lat_idx = np.argmin(np.abs(nc_lat - lat))
    
    # Extract currents (note: fields are [lat, lon])
    u = u_field[lat_idx, lon_idx]
    v = v_field[lat_idx, lon_idx]
    
    # Handle missing data
    if np.isnan(u):
        u = 0
    if np.isnan(v):
        v = 0
    
    return u, v


def visualize_drift_example(reef_data, nc_file='data/109516.nc'):
    """
    Show example of how currents affect larval drift from a source reef
    
    Returns:
        Dictionary with source location, drift path, and final location
    """
    # Pick a central reef as example
    example_reef = reef_data.iloc[len(reef_data)//2]
    
    with nc.Dataset(nc_file, 'r') as dataset:
        nc_lon = dataset.variables['longitude'][:]
        nc_lat = dataset.variables['latitude'][:]
        u_surface = dataset.variables['u_surface'][:]
        v_surface = dataset.variables['v_surface'][:]
        
        # Handle time dimension if present
        if len(u_surface.shape) == 3:
            # Average over time (first dimension)
            u_mean = np.mean(u_surface, axis=0)
            v_mean = np.mean(v_surface, axis=0)
        else:
            u_mean = u_surface
            v_mean = v_surface
    
    # Get current at source
    u, v = get_current_at_location(
        example_reef['Longitude'], 
        example_reef['Latitude'],
        u_mean, v_mean, nc_lon, nc_lat
    )
    
    # Calculate daily drift positions
    days = np.arange(0, 22)  # 0 to 21 days
    
    meters_per_degree_lon = 111000 * np.cos(np.radians(example_reef['Latitude']))
    meters_per_degree_lat = 111000
    
    u_deg_per_day = (u * 86400) / meters_per_degree_lon
    v_deg_per_day = (v * 86400) / meters_per_degree_lat
    
    drift_lons = example_reef['Longitude'] + (u_deg_per_day * days)
    drift_lats = example_reef['Latitude'] + (v_deg_per_day * days)
    
    return {
        'source_name': example_reef.get('SourceReef', 'Example Reef'),
        'source_lon': example_reef['Longitude'],
        'source_lat': example_reef['Latitude'],
        'current_u': u,
        'current_v': v,
        'current_speed': np.sqrt(u**2 + v**2),
        'drift_days': days.tolist(),
        'drift_lons': drift_lons.tolist(),
        'drift_lats': drift_lats.tolist(),
        'final_lon': drift_lons[-1],
        'final_lat': drift_lats[-1],
        'total_drift_km': np.sqrt(
            ((drift_lons[-1] - example_reef['Longitude']) * meters_per_degree_lon / 1000)**2 +
            ((drift_lats[-1] - example_reef['Latitude']) * meters_per_degree_lat / 1000)**2
        )
    }


if __name__ == "__main__":
    # Test the model
    print("Testing current-based dispersal model...")
    
    # Load reef data
    reef_metrics = pd.read_csv("output/st_marys/reef_metrics.csv")
    
    # Calculate settlement field with currents
    lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
        reef_metrics.iloc[:28]  # First 28 reefs
    )
    
    print(f"Settlement field shape: {settlement_prob.shape}")
    print(f"Max probability: {settlement_prob.max():.3f}")
    print(f"Mean probability: {settlement_prob.mean():.6f}")
    print(f"High settlement area (>0.5): {(settlement_prob > 0.5).sum()} grid cells")
    
    # Show drift example
    drift_info = visualize_drift_example(reef_metrics.iloc[:28])
    print(f"\nExample drift from {drift_info['source_name']}:")
    print(f"  Current speed: {drift_info['current_speed']:.3f} m/s")
    print(f"  Total drift over 21 days: {drift_info['total_drift_km']:.2f} km")
    print(f"  Direction: {'East' if drift_info['current_u'] > 0 else 'West'}-"
          f"{'North' if drift_info['current_v'] > 0 else 'South'}")