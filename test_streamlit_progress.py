#!/usr/bin/env python3
"""
Test script to verify Streamlit progress bar functionality
Run with: streamlit run test_streamlit_progress.py
"""

import streamlit as st
import pandas as pd
from python_dispersal_model import calculate_advection_diffusion_settlement

st.title("Testing Progress Bar in Settlement Map")

# Load reef data
reef_metrics = pd.read_csv("output/st_marys/reef_metrics.csv")
test_reefs = reef_metrics.iloc[:3]

st.write(f"Testing with {len(test_reefs)} reefs")

if st.button("Calculate Settlement Field"):
    # Create progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Define progress callback
    def update_progress(progress_pct, message):
        progress_bar.progress(progress_pct)
        status_text.text(message)
    
    # Calculate with progress updates
    st.info("Starting calculation...")
    
    lon_grid, lat_grid, settlement_prob = calculate_advection_diffusion_settlement(
        test_reefs,
        progress_callback=update_progress
    )
    
    # Clear progress indicators
    progress_bar.empty()
    status_text.empty()
    
    # Show results
    st.success("Calculation complete!")
    st.write(f"Grid size: {len(lon_grid)} x {len(lat_grid)}")
    st.write(f"Non-zero cells: {(settlement_prob > 0).sum():,}")
    st.write(f"Max probability: {settlement_prob.max():.3f}")
    
    # Simple visualization
    import plotly.graph_objects as go
    import numpy as np
    
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    fig = go.Figure(data=go.Heatmap(
        x=lon_grid,
        y=lat_grid,
        z=settlement_prob,
        colorscale='Viridis'
    ))
    
    fig.update_layout(
        title="Settlement Probability",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)