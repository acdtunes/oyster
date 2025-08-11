#!/usr/bin/env python3
"""
Realistic particle tracking using actual oceanographic data from NetCDF file.
Implements biophysical larval dispersal with real currents.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
import pandas as pd
from typing import Tuple, Dict, List
import warnings
warnings.filterwarnings('ignore')

class ParticleTracker:
    """Biophysical particle tracking with real ocean currents"""
    
    def __init__(self, nc_file_path: str = "data/109516.nc"):
        """Initialize with NetCDF data"""
        self.nc_file = nc_file_path
        self.data = None
        self.current_interpolators = {}
        self.load_oceanographic_data()
        
    def load_oceanographic_data(self):
        """Load and prepare ocean current data from NetCDF"""
        try:
            # Open NetCDF file
            self.data = xr.open_dataset(self.nc_file)
            
            # Extract dimensions
            self.lon = self.data.coords['longitude'].values if 'longitude' in self.data.coords else self.data.coords['lon'].values
            self.lat = self.data.coords['latitude'].values if 'latitude' in self.data.coords else self.data.coords['lat'].values
            
            # Current variables are u_surface and v_surface
            u_var = 'u_surface' if 'u_surface' in self.data.variables else None
            v_var = 'v_surface' if 'v_surface' in self.data.variables else None
                    
            if u_var and v_var:
                # Extract surface currents (monthly mean)
                # Using time index 5 (June) for spawning season
                time_idx = 5  # June
                self.u_current = self.data[u_var].isel(time=time_idx).values
                self.v_current = self.data[v_var].isel(time=time_idx).values
                
                # Handle NaN values
                self.u_current = np.nan_to_num(self.u_current, nan=0.0)
                self.v_current = np.nan_to_num(self.v_current, nan=0.0)
                
                # Scale currents to more realistic values for St. Mary's River
                # The mean should be around 0.05 m/s based on analysis
                current_speed = np.sqrt(self.u_current**2 + self.v_current**2)
                current_mean = np.mean(current_speed[current_speed > 0])
                
                # St. Mary's River has weak currents, typically 0.05 m/s
                target_mean = 0.05
                scale_factor = target_mean / current_mean if current_mean > 0 else 1.0
                self.u_current *= scale_factor
                self.v_current *= scale_factor
                print(f"Scaled currents by {scale_factor:.3f} (mean: {current_mean:.3f} -> {target_mean:.3f} m/s)")
                
                # Create interpolators for smooth particle movement
                self.current_interpolators['u'] = RegularGridInterpolator(
                    (self.lat, self.lon), 
                    self.u_current,
                    bounds_error=False,
                    fill_value=0.0
                )
                self.current_interpolators['v'] = RegularGridInterpolator(
                    (self.lat, self.lon),
                    self.v_current,
                    bounds_error=False,
                    fill_value=0.0
                )
                
                print(f"Loaded currents: u range [{np.min(self.u_current):.3f}, {np.max(self.u_current):.3f}] m/s")
                print(f"                v range [{np.min(self.v_current):.3f}, {np.max(self.v_current):.3f}] m/s")
                
            else:
                print(f"Warning: Could not find current variables. Available: {list(self.data.variables)}")
                # Use default weak currents
                self.setup_default_currents()
                
        except Exception as e:
            print(f"Error loading NetCDF: {e}")
            self.setup_default_currents()
    
    def setup_default_currents(self):
        """Setup default current field based on St. Mary's River characteristics"""
        # Create synthetic grid
        self.lon = np.linspace(-76.55, -76.33, 50)
        self.lat = np.linspace(38.09, 38.31, 50)
        
        # Weak eastward flow with some structure
        lon_grid, lat_grid = np.meshgrid(self.lon, self.lat)
        
        # Mean current 0.05 m/s eastward with spatial variation
        self.u_current = 0.05 + 0.02 * np.sin(2 * np.pi * (lon_grid + 76.44) / 0.1)
        self.v_current = 0.01 * np.cos(2 * np.pi * (lat_grid - 38.20) / 0.1)
        
        # Add some noise
        self.u_current += np.random.normal(0, 0.01, self.u_current.shape)
        self.v_current += np.random.normal(0, 0.01, self.v_current.shape)
        
        # Create interpolators
        self.current_interpolators['u'] = RegularGridInterpolator(
            (self.lat, self.lon), 
            self.u_current,
            bounds_error=False,
            fill_value=0.05
        )
        self.current_interpolators['v'] = RegularGridInterpolator(
            (self.lat, self.lon),
            self.v_current,
            bounds_error=False,
            fill_value=0.0
        )
        
    def get_current_at_location(self, lon: float, lat: float) -> Tuple[float, float]:
        """Get interpolated current velocity at a specific location"""
        point = np.array([[lat, lon]])
        u = self.current_interpolators['u'](point)[0]
        v = self.current_interpolators['v'](point)[0]
        return u, v
    
    def is_in_water(self, lon: float, lat: float) -> bool:
        """Check if location is in water (has valid current data)"""
        # For St. Mary's River, we'll use geographic bounds
        # The river is roughly in this box
        in_bounds = (-76.52 <= lon <= -76.35) and (38.12 <= lat <= 38.26)
        if not in_bounds:
            return False
        
        # Check if we have non-zero currents (indicates water)
        u, v = self.get_current_at_location(lon, lat)
        # Allow small currents, not just exactly 0
        return abs(u) > 0.001 or abs(v) > 0.001
    
    def simulate_particles(self, 
                          release_lon: float, 
                          release_lat: float,
                          n_particles: int = 100,
                          days: int = 21,
                          dt_hours: float = 1.0) -> Dict:
        """
        Simulate particle trajectories with real currents
        
        Parameters:
        -----------
        release_lon, release_lat : float
            Release location coordinates
        n_particles : int
            Number of particles to release
        days : int
            Simulation duration in days (PLD)
        dt_hours : float
            Time step in hours
            
        Returns:
        --------
        Dict with particle trajectories and statistics
        """
        
        # Parameters
        dt = dt_hours * 3600  # Convert to seconds
        n_steps = int(days * 24 / dt_hours)
        
        # Diffusion coefficient - reduced for estuarine environment
        D = 10.0  # m²/s (more appropriate for sheltered estuary)
        
        # Daily mortality rate (10%)
        daily_mortality = 0.1
        hourly_mortality = 1 - (1 - daily_mortality) ** (dt_hours / 24)
        
        # Initialize particle positions with small random spread
        particles = {
            'lon': np.random.normal(release_lon, 0.005, n_particles),
            'lat': np.random.normal(release_lat, 0.005, n_particles),
            'age': np.zeros(n_particles),
            'alive': np.ones(n_particles, dtype=bool),
            'trajectory_lon': [],
            'trajectory_lat': [],
            'trajectory_age': []
        }
        
        # Conversion factors
        meters_per_degree_lat = 111000  # meters
        meters_per_degree_lon = 111000 * np.cos(np.radians(release_lat))
        
        # Simulation loop
        for step in range(n_steps):
            
            # Store trajectories
            particles['trajectory_lon'].append(particles['lon'].copy())
            particles['trajectory_lat'].append(particles['lat'].copy())
            particles['trajectory_age'].append(particles['age'].copy())
            
            # Only update living particles
            alive_mask = particles['alive']
            n_alive = np.sum(alive_mask)
            
            if n_alive == 0:
                break
                
            # Get currents at particle locations
            u_velocities = np.zeros(n_particles)
            v_velocities = np.zeros(n_particles)
            
            for i in range(n_particles):
                if alive_mask[i]:
                    u, v = self.get_current_at_location(
                        particles['lon'][i], 
                        particles['lat'][i]
                    )
                    u_velocities[i] = u
                    v_velocities[i] = v
            
            # Advection (Euler forward for simplicity)
            particles['lon'][alive_mask] += (u_velocities[alive_mask] * dt / meters_per_degree_lon)
            particles['lat'][alive_mask] += (v_velocities[alive_mask] * dt / meters_per_degree_lat)
            
            # Diffusion (random walk)
            diffusion_std_lon = np.sqrt(2 * D * dt) / meters_per_degree_lon
            diffusion_std_lat = np.sqrt(2 * D * dt) / meters_per_degree_lat
            
            particles['lon'][alive_mask] += np.random.normal(0, diffusion_std_lon, n_alive)
            particles['lat'][alive_mask] += np.random.normal(0, diffusion_std_lat, n_alive)
            
            # Apply mortality
            mortality_roll = np.random.random(n_particles)
            particles['alive'] = particles['alive'] & (mortality_roll > hourly_mortality)
            
            # Update age
            particles['age'] += dt_hours / 24  # Convert to days
            
            # Boundary checking - St. Mary's River approximate bounds
            # The river is roughly between -76.55 to -76.33 lon and 38.09 to 38.31 lat
            # But we need tighter bounds for the actual river channel
            particles['lon'] = np.clip(particles['lon'], -76.52, -76.35)
            particles['lat'] = np.clip(particles['lat'], 38.12, 38.26)
            
            # Simple boundary check - only kill particles that go way out of bounds
            # St. Mary's River is a relatively small area
            far_out = (
                (particles['lon'] < -76.55) | (particles['lon'] > -76.30) |
                (particles['lat'] < 38.10) | (particles['lat'] > 38.30)
            )
            particles['alive'][far_out] = False
        
        # Convert trajectories to arrays
        particles['trajectory_lon'] = np.array(particles['trajectory_lon'])
        particles['trajectory_lat'] = np.array(particles['trajectory_lat'])
        particles['trajectory_age'] = np.array(particles['trajectory_age'])
        
        # Calculate statistics
        stats = {
            'survival_rate': np.sum(particles['alive']) / n_particles,
            'mean_displacement_km': np.mean([
                self.haversine_distance(release_lon, release_lat, 
                                       particles['lon'][i], particles['lat'][i])
                for i in range(n_particles) if particles['alive'][i]
            ]) if np.any(particles['alive']) else 0,
            'max_displacement_km': np.max([
                self.haversine_distance(release_lon, release_lat,
                                       particles['lon'][i], particles['lat'][i])
                for i in range(n_particles) if particles['alive'][i]
            ]) if np.any(particles['alive']) else 0,
            'n_alive': np.sum(particles['alive'])
        }
        
        return particles, stats
    
    @staticmethod
    def haversine_distance(lon1, lat1, lon2, lat2):
        """Calculate distance between two points in km"""
        R = 6371  # Earth radius in km
        
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
        a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_current_field_subset(self, lon_min=-76.50, lon_max=-76.35, 
                                 lat_min=38.15, lat_max=38.25):
        """Extract current field for visualization"""
        lon_mask = (self.lon >= lon_min) & (self.lon <= lon_max)
        lat_mask = (self.lat >= lat_min) & (self.lat <= lat_max)
        
        return {
            'lon': self.lon[lon_mask],
            'lat': self.lat[lat_mask],
            'u': self.u_current[np.ix_(lat_mask, lon_mask)],
            'v': self.v_current[np.ix_(lat_mask, lon_mask)]
        }


# Testing function
if __name__ == "__main__":
    print("Initializing Particle Tracker with real ocean data...")
    tracker = ParticleTracker()
    
    print("\nRunning test simulation...")
    # St. Mary's River approximate center
    particles, stats = tracker.simulate_particles(
        release_lon=-76.44,
        release_lat=38.19,
        n_particles=100,
        days=21
    )
    
    print(f"\nSimulation Results:")
    print(f"  Survival rate: {stats['survival_rate']:.1%}")
    print(f"  Mean displacement: {stats['mean_displacement_km']:.1f} km")
    print(f"  Max displacement: {stats['max_displacement_km']:.1f} km")
    print(f"  Particles alive: {stats['n_alive']}/{100}")