"""
Data loading module for oyster dispersal analysis
"""
import pandas as pd
import numpy as np
import netCDF4 as nc
import streamlit as st

@st.cache_data
def load_connectivity_data():
    """Load connectivity matrix and reef metrics"""
    try:
        conn_df = pd.read_csv("output/st_marys/connectivity_matrix.csv", index_col=0)
        reef_metrics = pd.read_csv("output/st_marys/reef_metrics.csv")
        
        # Ensure the matrix is square
        n_reefs = min(conn_df.shape[0], conn_df.shape[1], len(reef_metrics))
        
        # Create a square matrix
        conn_matrix = np.zeros((n_reefs, n_reefs))
        for i in range(min(n_reefs, conn_df.shape[0])):
            for j in range(min(n_reefs, conn_df.shape[1])):
                conn_matrix[i, j] = conn_df.iloc[i, j]
        
        return conn_matrix, reef_metrics, n_reefs
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

@st.cache_data
def load_netcdf_data():
    """Load NetCDF oceanographic data"""
    try:
        with nc.Dataset('data/109516.nc', 'r') as dataset:
            lon = dataset.variables['longitude'][:]
            lat = dataset.variables['latitude'][:]
            u_surface = dataset.variables['u_surface'][:]
            v_surface = dataset.variables['v_surface'][:]
            mask = dataset.variables['mask_land_sea'][:]
            
            # Average over time if 3D
            if len(u_surface.shape) == 3:
                u_mean = np.mean(u_surface, axis=0)
                v_mean = np.mean(v_surface, axis=0)
            else:
                u_mean = u_surface
                v_mean = v_surface
                
            return {
                'lon': lon,
                'lat': lat,
                'u_mean': u_mean,
                'v_mean': v_mean,
                'mask': mask
            }
    except Exception as e:
        st.error(f"Error loading NetCDF data: {e}")
        return None

def get_reef_subset(reef_metrics, n_reefs=None):
    """Get a subset of reef data"""
    if n_reefs is None:
        n_reefs = min(28, len(reef_metrics))
    return reef_metrics.iloc[:n_reefs]