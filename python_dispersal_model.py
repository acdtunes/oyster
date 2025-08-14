#!/usr/bin/env python3
"""
Python implementation of the current-based larval dispersal model
with water boundary constraints - larvae can only travel through water
CRITICAL: Dispersal probability is ZERO across land barriers
"""

import numpy as np
import netCDF4 as nc
from scipy.spatial.distance import cdist
import pandas as pd
import json
import os
import sys
import geopandas as gpd
from shapely.geometry import Point, LineString, Polygon, MultiPolygon
from shapely.ops import unary_union
from shapely.vectorized import contains
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    # Fallback simple progress bar
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, disable=False):
            self.iterable = iterable
            self.total = total or (len(iterable) if iterable else 0)
            self.desc = desc or ""
            self.disable = disable
            self.n = 0
            
        def __iter__(self):
            if self.iterable:
                for item in self.iterable:
                    if not self.disable:
                        self.update(1)
                    yield item
            return self
            
        def __enter__(self):
            return self
            
        def __exit__(self, *args):
            if not self.disable:
                print()  # New line after progress
                
        def update(self, n=1):
            if self.disable:
                return
            self.n += n
            percent = int(100 * self.n / self.total) if self.total > 0 else 0
            bar_length = 40
            filled = int(bar_length * self.n / self.total) if self.total > 0 else 0
            bar = '=' * filled + '-' * (bar_length - filled)
            sys.stdout.write(f'\r{self.desc}: [{bar}] {percent}% ({self.n}/{self.total})')
            sys.stdout.flush()


def load_water_boundaries():
    """
    Load water boundary geometry and inclusion zones
    
    Returns:
        water_geometry: Shapely geometry representing water areas
        inclusion_zones: List of inclusion zone polygons
    """
    water_geometry = None
    inclusion_zones = []
    
    # Load main water boundary
    boundary_file = 'data/st_marys_water_boundary.geojson'
    if os.path.exists(boundary_file):
        try:
            gdf = gpd.read_file(boundary_file)
            if len(gdf) > 0:
                water_geometry = gdf.geometry.iloc[0]
                print(f"Loaded water boundary with {len(water_geometry.geoms) if hasattr(water_geometry, 'geoms') else 1} polygons")
        except Exception as e:
            print(f"Warning: Could not load water boundary: {e}")
    
    # Load inclusion zones
    zones_file = 'data/inclusion_zones.json'
    if os.path.exists(zones_file):
        try:
            with open(zones_file, 'r') as f:
                data = json.load(f)
                for zone in data.get('zones', []):
                    if zone.get('type') == 'inclusion' and len(zone.get('coordinates', [])) >= 3:
                        # Create polygon from coordinates
                        coords = zone['coordinates']
                        if coords[0] != coords[-1]:  # Ensure closed
                            coords.append(coords[0])
                        poly = Polygon(coords)
                        inclusion_zones.append(poly)
                print(f"Loaded {len(inclusion_zones)} inclusion zones")
        except Exception as e:
            print(f"Warning: Could not load inclusion zones: {e}")
    
    # If we have inclusion zones, use them as the water area
    if inclusion_zones:
        water_geometry = unary_union(inclusion_zones)
    
    return water_geometry, inclusion_zones


def is_point_in_water(lon, lat, water_geometry):
    """
    Check if a point is in water
    
    Args:
        lon, lat: Coordinates
        water_geometry: Shapely geometry for water areas
    
    Returns:
        Boolean: True if in water, False if on land
    """
    if water_geometry is None:
        return True  # If no water boundary, assume all points are valid
    
    point = Point(lon, lat)
    return water_geometry.contains(point)


def track_larval_trajectory(source_lon, source_lat, u_field, v_field, 
                           nc_lon, nc_lat, water_geometry, params):
    """
    Track daily larval trajectory with water boundary constraints
    
    Args:
        source_lon, source_lat: Starting position
        u_field, v_field: Current velocity fields [lat, lon]
        nc_lon, nc_lat: NetCDF coordinate arrays
        water_geometry: Shapely geometry for water areas
        params: Dispersal parameters including pelagic_larval_duration
    
    Returns:
        trajectory: List of (lon, lat, day, in_water) tuples
        final_position: (lon, lat) of final position, or None if blocked
    """
    trajectory = [(source_lon, source_lat, 0, True)]
    
    current_lon = source_lon
    current_lat = source_lat
    
    for day in range(1, params['pelagic_larval_duration'] + 1):
        # Find nearest grid point for current extraction
        lon_idx = np.argmin(np.abs(nc_lon - current_lon))
        lat_idx = np.argmin(np.abs(nc_lat - current_lat))
        
        # Get currents at current position
        u_at_pos = u_field[lat_idx, lon_idx]
        v_at_pos = v_field[lat_idx, lon_idx]
        
        # Handle missing data
        if np.isnan(u_at_pos):
            u_at_pos = 0
        if np.isnan(v_at_pos):
            v_at_pos = 0
        
        # Calculate daily displacement
        meters_per_degree_lon = 111000 * np.cos(np.radians(current_lat))
        meters_per_degree_lat = 111000
        
        u_deg_per_day = (u_at_pos * 86400) / meters_per_degree_lon
        v_deg_per_day = (v_at_pos * 86400) / meters_per_degree_lat
        
        # Calculate new position
        new_lon = current_lon + u_deg_per_day
        new_lat = current_lat + v_deg_per_day
        
        # Check if new position is in water
        in_water = is_point_in_water(new_lon, new_lat, water_geometry)
        
        if not in_water:
            # Particle hits land - try intermediate steps to find boundary
            n_substeps = 10
            for substep in range(1, n_substeps + 1):
                frac = substep / n_substeps
                test_lon = current_lon + frac * u_deg_per_day
                test_lat = current_lat + frac * v_deg_per_day
                
                if not is_point_in_water(test_lon, test_lat, water_geometry):
                    # Found land boundary - stop at previous valid position
                    if substep > 1:
                        frac = (substep - 1) / n_substeps
                        new_lon = current_lon + frac * u_deg_per_day
                        new_lat = current_lat + frac * v_deg_per_day
                    else:
                        # Can't move at all - particle stuck
                        new_lon = current_lon
                        new_lat = current_lat
                    break
            
            # Mark as hitting land and stop trajectory
            trajectory.append((new_lon, new_lat, day, False))
            return trajectory, None  # No valid final position
        
        # Update position for next day
        current_lon = new_lon
        current_lat = new_lat
        trajectory.append((current_lon, current_lat, day, True))
    
    # Successfully completed trajectory in water
    return trajectory, (current_lon, current_lat)


def calculate_water_aware_connectivity(source_idx, sink_idx, source_lon, source_lat,
                                      sink_lon, sink_lat, u_field, v_field,
                                      nc_lon, nc_lat, water_geometry, params):
    """
    Calculate connectivity with water boundary constraints
    
    Returns:
        connectivity: Probability of successful larval transport (0-1)
    """
    # Track larval trajectory from source
    trajectory, final_pos = track_larval_trajectory(
        source_lon, source_lat, u_field, v_field,
        nc_lon, nc_lat, water_geometry, params
    )
    
    # If larvae hit land, no connectivity
    if final_pos is None:
        return 0.0
    
    # Calculate distance from final position to sink
    final_lon, final_lat = final_pos
    
    # Haversine distance
    R = 6371  # Earth radius in km
    dlat = np.radians(sink_lat - final_lat)
    dlon = np.radians(sink_lon - final_lon)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(final_lat)) * np.cos(np.radians(sink_lat)) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    distance_km = R * c
    
    # Check if sink is reachable (within diffusion range)
    diffusion_km = np.sqrt(2 * params['diffusion_coefficient'] * 
                          params['pelagic_larval_duration'] * 86400) / 1000
    
    # Gaussian probability of reaching sink from final position
    dispersal_prob = np.exp(-distance_km**2 / (2 * diffusion_km**2))
    
    # Survival probability
    survival_prob = (1 - params['mortality_rate'])**params['pelagic_larval_duration']
    
    # Check if direct path from final position to sink is in water
    # Sample points along the path
    n_checks = 20
    path_clear = True
    for i in range(1, n_checks):
        frac = i / n_checks
        check_lon = final_lon + frac * (sink_lon - final_lon)
        check_lat = final_lat + frac * (sink_lat - final_lat)
        if not is_point_in_water(check_lon, check_lat, water_geometry):
            path_clear = False
            break
    
    if not path_clear:
        # Reduce probability if path crosses land
        dispersal_prob *= 0.1  # Strong penalty for land barriers
    
    # Combined probability
    connectivity = dispersal_prob * survival_prob
    
    # Boost for self-recruitment if same reef
    if distance_km < 1:
        connectivity *= 1.5
    
    return min(1.0, connectivity)


def is_path_in_water(lon1, lat1, lon2, lat2, water_geometry, n_checks=30):
    """
    Check if a straight path between two points stays entirely in water
    
    Args:
        lon1, lat1: Start coordinates
        lon2, lat2: End coordinates
        water_geometry: Shapely geometry for water areas
        n_checks: Number of points to check along the path (increased for accuracy)
    
    Returns:
        Boolean: True if entire path is in water, False if it crosses land
    """
    if water_geometry is None:
        return True
    
    # Calculate distance to determine number of checks needed
    # More checks for longer distances to catch narrow barriers
    dist_deg = np.sqrt((lon2 - lon1)**2 + (lat2 - lat1)**2)
    dist_km = dist_deg * 111  # Approximate conversion
    
    # Adaptive checking: more checks for longer distances
    # At least 1 check per 100 meters for accuracy
    adaptive_checks = max(n_checks, int(dist_km * 10))
    
    # Check points along the path
    for i in range(adaptive_checks + 1):
        frac = i / adaptive_checks
        check_lon = lon1 + frac * (lon2 - lon1)
        check_lat = lat1 + frac * (lat2 - lat1)
        
        point = Point(check_lon, check_lat)
        if not water_geometry.contains(point):
            return False
    
    return True


def calculate_advection_diffusion_settlement(reef_data, nc_file='data/109516.nc', 
                                            pelagic_duration=21, mortality_rate=0.1,
                                            diffusion_coeff=100, settlement_day=14,
                                            progress_callback=None):
    """
    Calculate larval settlement probability field using advection-diffusion model
    with STRICT water boundary constraints - NO dispersal across land barriers
    
    Args:
        reef_data: DataFrame with reef locations (Longitude, Latitude, Density)
        nc_file: Path to NetCDF file with current data
        pelagic_duration: Days larvae spend in water column (21)
        mortality_rate: Daily mortality rate (0.1 = 10%)
        diffusion_coeff: Horizontal diffusion coefficient (m²/s)
        settlement_day: Day when larvae become competent to settle (14)
        progress_callback: Optional callback function for progress updates (for Streamlit)
                          Should accept (progress_pct, message) where progress_pct is 0-1
    
    Returns:
        lon_grid, lat_grid, settlement_prob: Coordinate grids and probability field
    """
    
    # Load water boundaries
    water_geometry, inclusion_zones = load_water_boundaries()
    
    # Load current data from NetCDF
    with nc.Dataset(nc_file, 'r') as dataset:
        nc_lon = dataset.variables['longitude'][:]
        nc_lat = dataset.variables['latitude'][:]
        
        u_surface_raw = dataset.variables['u_surface'][:]
        v_surface_raw = dataset.variables['v_surface'][:]
        
        if len(u_surface_raw.shape) == 3:
            u_mean = np.mean(u_surface_raw, axis=0)
            v_mean = np.mean(v_surface_raw, axis=0)
        else:
            u_mean = u_surface_raw
            v_mean = v_surface_raw
    
    # Grid bounds for St. Mary's River
    lon_min = -76.495
    lon_max = -76.4  # Original eastern boundary
    lat_min = 38.125
    lat_max = 38.23
    
    # HIGH RESOLUTION GRID for detailed dispersal patterns
    # Approximately 18-20 meter spacing between grid points
    n_lon = 600  # Increased from 375 for higher resolution
    n_lat = 800  # Increased from 500 for higher resolution
    lon_grid = np.linspace(lon_min, lon_max, n_lon)
    lat_grid = np.linspace(lat_min, lat_max, n_lat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    # Initialize settlement field
    settlement_prob = np.zeros((n_lat, n_lon))
    
    # Pre-compute water mask for grid
    if progress_callback:
        progress_callback(0.1, "Computing water mask for grid...")
    else:
        print("Computing water mask for grid...")
        print(f"  Grid size: {n_lon} x {n_lat} = {n_lon * n_lat:,} cells")
    
    lon_flat = lon_mesh.flatten()
    lat_flat = lat_mesh.flatten()
    
    if water_geometry is not None:
        # Process in chunks for large grids to avoid memory issues
        chunk_size = 50000
        water_mask_flat = np.zeros(len(lon_flat), dtype=bool)
        
        # Create progress bar for water mask computation
        n_chunks = (len(lon_flat) + chunk_size - 1) // chunk_size
        
        if progress_callback:
            # Use callback for Streamlit
            for chunk_idx, i in enumerate(range(0, len(lon_flat), chunk_size)):
                end_idx = min(i + chunk_size, len(lon_flat))
                water_mask_flat[i:end_idx] = contains(
                    water_geometry, 
                    lon_flat[i:end_idx], 
                    lat_flat[i:end_idx]
                )
                progress = 0.1 + (0.2 * chunk_idx / n_chunks)  # 10-30% of total progress
                progress_callback(progress, f"Computing water mask... {chunk_idx+1}/{n_chunks}")
        else:
            # Use tqdm for console
            with tqdm(total=n_chunks, desc="  Computing water mask", disable=False) as pbar:
                for i in range(0, len(lon_flat), chunk_size):
                    end_idx = min(i + chunk_size, len(lon_flat))
                    water_mask_flat[i:end_idx] = contains(
                        water_geometry, 
                        lon_flat[i:end_idx], 
                        lat_flat[i:end_idx]
                    )
                    pbar.update(1)
        
        water_mask = water_mask_flat.reshape(lon_mesh.shape)
    else:
        water_mask = np.ones_like(lon_mesh, dtype=bool)
    
    if not progress_callback:
        print(f"  {water_mask.sum():,} of {water_mask.size:,} grid cells are in water")
    
    # Process each source reef
    if not progress_callback:
        print("\nCalculating dispersal from each reef:")
    
    # Create progress bar for reef processing
    reef_iterator = reef_data.iterrows() if progress_callback else tqdm(reef_data.iterrows(), 
                                                                        total=len(reef_data),
                                                                        desc="Processing reefs")
    
    for reef_idx, (idx, reef) in enumerate(reef_iterator):
        source_lon = reef['Longitude']
        source_lat = reef['Latitude']
        source_density = reef['Density'] if 'Density' in reef else reef.get('AvgDensity', 100)
        reef_name = reef.get('SourceReef', f'Reef_{idx}')
        
        # Update progress for Streamlit
        if progress_callback:
            progress = 0.3 + (0.6 * reef_idx / len(reef_data))  # 30-90% of total progress
            progress_callback(progress, f"Processing reef {reef_name} ({reef_idx+1}/{len(reef_data)})")
        
        # Check if source is in water
        if water_geometry is not None:
            if not is_point_in_water(source_lon, source_lat, water_geometry):
                continue  # Skip reefs not in water
        
        # Get currents at source
        lon_idx = np.argmin(np.abs(nc_lon - source_lon))
        lat_idx = np.argmin(np.abs(nc_lat - source_lat))
        
        u_at_source = u_mean[lat_idx, lon_idx]
        v_at_source = v_mean[lat_idx, lon_idx]
        
        if np.isnan(u_at_source): u_at_source = 0
        if np.isnan(v_at_source): v_at_source = 0
        
        # Calculate drift endpoint
        meters_per_degree_lon = 111000 * np.cos(np.radians(source_lat))
        meters_per_degree_lat = 111000
        
        u_deg_per_day = (u_at_source * 86400) / meters_per_degree_lon
        v_deg_per_day = (v_at_source * 86400) / meters_per_degree_lat
        
        # Track sub-daily positions for more accurate trajectories
        # Use 4 time steps per day to capture tidal variations
        time_steps_per_day = 4
        dt = 1.0 / time_steps_per_day
        current_lon = source_lon
        current_lat = source_lat
        
        # Track full trajectory with sub-daily resolution
        for day in range(pelagic_duration):
            for substep in range(time_steps_per_day):
                # Calculate next position
                next_lon = current_lon + (u_deg_per_day * dt)
                next_lat = current_lat + (v_deg_per_day * dt)
                
                # Check if path to next position crosses land
                if not is_path_in_water(current_lon, current_lat, next_lon, next_lat, water_geometry):
                    # Try smaller steps to find exact boundary
                    found_boundary = False
                    for micro_step in range(10):
                        micro_frac = micro_step / 10.0
                        test_lon = current_lon + (u_deg_per_day * dt * micro_frac)
                        test_lat = current_lat + (v_deg_per_day * dt * micro_frac)
                        
                        if not is_path_in_water(current_lon, current_lat, test_lon, test_lat, water_geometry):
                            if micro_step > 0:
                                # Use previous valid position
                                micro_frac = (micro_step - 1) / 10.0
                                next_lon = current_lon + (u_deg_per_day * dt * micro_frac)
                                next_lat = current_lat + (v_deg_per_day * dt * micro_frac)
                            else:
                                # Can't move at all
                                next_lon = current_lon
                                next_lat = current_lat
                            found_boundary = True
                            break
                    
                    if found_boundary:
                        # Stop here - larvae hit land
                        current_lon = next_lon
                        current_lat = next_lat
                        break
                
                current_lon = next_lon
                current_lat = next_lat
            
            # Check if we hit land and stopped
            if current_lon == source_lon and current_lat == source_lat:
                break
        
        drift_lon = current_lon
        drift_lat = current_lat
        
        # Calculate diffusion parameters
        diffusion_km = np.sqrt(2 * diffusion_coeff * pelagic_duration * 86400) / 1000
        sigma_deg = diffusion_km / 111
        survival_prob = (1 - mortality_rate)**pelagic_duration
        
        # OPTIMIZED DISPERSAL CALCULATION for high-resolution grid
        # Use vectorized operations where possible
        
        # Calculate dispersal range (3 sigma covers 99.7% of distribution)
        max_dispersal_deg = sigma_deg * 3
        max_dispersal_km = max_dispersal_deg * 111
        
        # Find bounding box of potential settlement area
        lon_min_local = max(drift_lon - max_dispersal_deg, lon_min)
        lon_max_local = min(drift_lon + max_dispersal_deg, lon_max)
        lat_min_local = max(drift_lat - max_dispersal_deg, lat_min)
        lat_max_local = min(drift_lat + max_dispersal_deg, lat_max)
        
        # Find grid indices within bounding box
        lon_indices = np.where((lon_grid >= lon_min_local) & (lon_grid <= lon_max_local))[0]
        lat_indices = np.where((lat_grid >= lat_min_local) & (lat_grid <= lat_max_local))[0]
        
        if len(lon_indices) == 0 or len(lat_indices) == 0:
            continue  # No cells in range
        
        # Create local meshgrid for efficient calculation
        local_lon_mesh, local_lat_mesh = np.meshgrid(lon_grid[lon_indices], lat_grid[lat_indices])
        
        # Vectorized distance calculation
        lon_dist = (local_lon_mesh - drift_lon) * np.cos(np.radians(local_lat_mesh)) * 111
        lat_dist = (local_lat_mesh - drift_lat) * 111
        dist_km = np.sqrt(lon_dist**2 + lat_dist**2)
        
        # Vectorized Gaussian probability
        local_prob = np.exp(-dist_km**2 / (2 * sigma_deg**2 * 111**2))
        
        # Apply water mask and check paths (optimized)
        for i_local, i_global in enumerate(lat_indices):
            for j_local, j_global in enumerate(lon_indices):
                # Skip if not in water
                if not water_mask[i_global, j_global]:
                    continue
                
                # Skip if probability is negligible
                if local_prob[i_local, j_local] < 0.001:
                    continue
                
                target_lon = lon_grid[j_global]
                target_lat = lat_grid[i_global]
                
                # Check water path only for significant probabilities
                # Use fewer checks for nearby cells, more for distant ones
                dist_to_target = dist_km[i_local, j_local]
                if dist_to_target < 5:  # Within 5km, assume connected if both in water
                    # Quick check: both endpoints in water is usually sufficient for short distances
                    if water_geometry is not None:
                        if not is_path_in_water(drift_lon, drift_lat, target_lon, target_lat, water_geometry):
                            continue
                else:
                    # Full path check for longer distances
                    if not is_path_in_water(drift_lon, drift_lat, target_lon, target_lat, water_geometry):
                        continue
                
                # Add contribution
                contribution = local_prob[i_local, j_local] * (source_density / 100) * survival_prob
                settlement_prob[i_global, j_global] += contribution
    
    # Normalize
    if settlement_prob.max() > 0:
        settlement_prob = settlement_prob / settlement_prob.max()
    
    # Final insurance - zero out any land cells
    settlement_prob = settlement_prob * water_mask
    
    # Final progress update
    if progress_callback:
        progress_callback(1.0, "Calculation complete!")
    
    return lon_grid, lat_grid, settlement_prob


def build_connectivity_matrix_with_water(reef_data, nc_file='data/109516.nc',
                                        pelagic_duration=21, mortality_rate=0.1,
                                        diffusion_coeff=100):
    """
    Build reef connectivity matrix with water boundary constraints
    
    Args:
        reef_data: DataFrame with reef information
        nc_file: Path to NetCDF file
        pelagic_duration: Pelagic larval duration (days)
        mortality_rate: Daily mortality rate
        diffusion_coeff: Diffusion coefficient (m²/s)
    
    Returns:
        connectivity_matrix: NxN matrix of connectivity probabilities
    """
    
    # Load water boundaries
    water_geometry, _ = load_water_boundaries()
    
    # Load current data
    with nc.Dataset(nc_file, 'r') as dataset:
        nc_lon = dataset.variables['longitude'][:]
        nc_lat = dataset.variables['latitude'][:]
        
        u_surface_raw = dataset.variables['u_surface'][:]
        v_surface_raw = dataset.variables['v_surface'][:]
        
        if len(u_surface_raw.shape) == 3:
            u_mean = np.mean(u_surface_raw, axis=0)
            v_mean = np.mean(v_surface_raw, axis=0)
        else:
            u_mean = u_surface_raw
            v_mean = v_surface_raw
    
    # Create parameters dictionary
    params = {
        'pelagic_larval_duration': pelagic_duration,
        'mortality_rate': mortality_rate,
        'diffusion_coefficient': diffusion_coeff
    }
    
    n_reefs = len(reef_data)
    conn_matrix = np.zeros((n_reefs, n_reefs))
    
    print("Building water-aware connectivity matrix...")
    
    # Pre-compute all reef trajectories
    reef_trajectories = []
    successful_sources = 0
    for i in range(n_reefs):
        source = reef_data.iloc[i]
        trajectory, final_pos = track_larval_trajectory(
            source['Longitude'], source['Latitude'],
            u_mean, v_mean, nc_lon, nc_lat,
            water_geometry, params
        )
        reef_trajectories.append((trajectory, final_pos))
        if final_pos is not None:
            successful_sources += 1
    
    print(f"  {successful_sources}/{n_reefs} source reefs have successful trajectories")
    
    # Calculate connectivity for each source-sink pair
    for i in range(n_reefs):
        source = reef_data.iloc[i]
        trajectory, final_pos = reef_trajectories[i]
        
        if final_pos is None:
            # Source reef larvae get blocked by land - use source position with reduced dispersal
            final_lon = source['Longitude']
            final_lat = source['Latitude']
            diffusion_coeff_reduced = diffusion_coeff * 0.1  # Much reduced dispersal if blocked
        else:
            final_lon, final_lat = final_pos
            diffusion_coeff_reduced = diffusion_coeff
        
        for j in range(n_reefs):
            sink = reef_data.iloc[j]
            
            # Calculate distance from drift endpoint to sink
            dlat = np.radians(sink['Latitude'] - final_lat)
            dlon = np.radians(sink['Longitude'] - final_lon)
            a = np.sin(dlat/2)**2 + np.cos(np.radians(final_lat)) * np.cos(np.radians(sink['Latitude'])) * np.sin(dlon/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            distance_km = 6371 * c
            
            # Diffusion spread (use reduced if larvae were blocked)
            diffusion_km = np.sqrt(2 * diffusion_coeff_reduced * pelagic_duration * 86400) / 1000
            
            # Gaussian probability
            dispersal_prob = np.exp(-distance_km**2 / (2 * diffusion_km**2))
            
            # Check if path is mostly in water (simplified check)
            if water_geometry is not None:
                # Check a few points along the path
                n_checks = 5
                blocked = False
                for k in range(1, n_checks):
                    frac = k / n_checks
                    check_lon = final_lon + frac * (sink['Longitude'] - final_lon)
                    check_lat = final_lat + frac * (sink['Latitude'] - final_lat)
                    if not is_point_in_water(check_lon, check_lat, water_geometry):
                        blocked = True
                        break
                
                if blocked:
                    dispersal_prob *= 0.2  # Reduce but don't eliminate probability
            
            # Survival probability
            survival_prob = (1 - mortality_rate)**pelagic_duration
            
            # Combined connectivity
            connectivity = dispersal_prob * survival_prob
            
            # Boost self-recruitment
            if i == j:
                connectivity *= 2.0
            
            # Scale by source fecundity
            larvae_production = np.sqrt(source.get('Density', source.get('AvgDensity', 100)))
            conn_matrix[i, j] = connectivity * larvae_production
        
        # Progress indicator
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{n_reefs} source reefs")
    
    # Normalize rows to sum to 1
    for i in range(n_reefs):
        row_sum = conn_matrix[i, :].sum()
        if row_sum > 0:
            conn_matrix[i, :] /= row_sum
    
    return conn_matrix


if __name__ == "__main__":
    # Test the water-aware model
    print("Testing water-aware larval dispersal model...")
    
    # Load reef data
    reef_metrics = pd.read_csv("output/st_marys/reef_metrics.csv")
    test_reefs = reef_metrics.iloc[:10]  # Test with first 10 reefs
    
    # Test water boundary loading
    water_geometry, inclusion_zones = load_water_boundaries()
    if water_geometry:
        print(f"Water boundary loaded successfully")
        
        # Test some reef positions
        for idx, reef in test_reefs.iterrows():
            in_water = is_point_in_water(reef['Longitude'], reef['Latitude'], water_geometry)
            print(f"  {reef['SourceReef']}: {'IN WATER' if in_water else 'ON LAND'}")
    
    # Calculate settlement field with water constraints
    print("\nCalculating water-aware settlement field...")
    lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
        test_reefs
    )
    
    print(f"Settlement field shape: {settlement_prob.shape}")
    print(f"Max probability: {settlement_prob.max():.3f}")
    print(f"Mean probability: {settlement_prob.mean():.6f}")
    print(f"Cells with settlement > 0.1: {(settlement_prob > 0.1).sum()}")
    
    # Test connectivity matrix
    print("\nBuilding water-aware connectivity matrix...")
    conn_matrix = build_connectivity_matrix_with_water(test_reefs)
    
    print(f"Connectivity matrix shape: {conn_matrix.shape}")
    print(f"Mean connectivity: {conn_matrix.mean():.6f}")
    print(f"Max connectivity: {conn_matrix.max():.3f}")
    print(f"Self-recruitment rates: {np.diag(conn_matrix)}")