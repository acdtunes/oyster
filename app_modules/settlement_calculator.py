"""
Cached settlement field calculator to avoid repeated expensive computations
"""
import numpy as np
import streamlit as st
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

@st.cache_data
def get_settlement_field(n_reefs=28):
    """
    Calculate settlement field ONCE and cache it
    This is expensive so we only want to do it once per session
    """
    try:
        # Load reef data
        import pandas as pd
        reef_metrics = pd.read_csv('output/st_marys/reef_metrics.csv')
        reef_data = reef_metrics.iloc[:n_reefs]
        
        from python_dispersal_model import calculate_advection_diffusion_settlement
        
        # Calculate current-based settlement field
        lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
            reef_data,
            nc_file='data/109516.nc',
            pelagic_duration=21,
            mortality_rate=0.1,
            diffusion_coeff=100,
            settlement_day=14
        )
        
        return lon_grid, lat_grid, settlement_prob, reef_data
        
    except Exception as e:
        # Fallback to simple model
        print(f"Using fallback model: {e}")
        
        # Simple fallback
        import pandas as pd
        reef_metrics = pd.read_csv('output/st_marys/reef_metrics.csv')
        reef_data = reef_metrics.iloc[:n_reefs]
        
        lon_grid = np.linspace(-76.495, -76.4, 100)
        lat_grid = np.linspace(38.125, 38.23, 100)
        lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
        
        settlement_prob = np.zeros_like(lon_mesh)
        for _, reef in reef_data.iterrows():
            lon_dist = (lon_mesh - reef['Longitude']) * 111 * np.cos(np.radians(reef['Latitude']))
            lat_dist = (lat_mesh - reef['Latitude']) * 111
            dist_km = np.sqrt(lon_dist**2 + lat_dist**2)
            settlement_prob += np.exp(-dist_km**2 / 8) * (reef.get('Density', 100) / 100)
        
        if settlement_prob.max() > 0:
            settlement_prob = settlement_prob / settlement_prob.max()
        
        return lon_grid, lat_grid, settlement_prob, reef_data

@st.cache_data
def get_water_boundary():
    """Load water boundary once and cache it"""
    try:
        from usgs_data_download import load_water_boundary_data
        water_geometry = load_water_boundary_data('st_marys_water_boundary.geojson')
        return water_geometry
    except:
        return None