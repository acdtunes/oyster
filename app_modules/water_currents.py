"""
Water Currents visualization module
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from app_modules.data_loader import load_netcdf_data

def create_visualization(reef_metrics):
    """Create water current visualization"""
    nc_data = load_netcdf_data()
    
    if nc_data is None:
        return None
    
    lon = nc_data['lon']
    lat = nc_data['lat']
    u_mean = nc_data['u_mean']
    v_mean = nc_data['v_mean']
    mask = nc_data['mask']
    
    # Get reef bounds
    n_reefs = min(28, len(reef_metrics))
    reef_data = reef_metrics.iloc[:n_reefs]
    
    lon_min = reef_data['Longitude'].min() - 0.05
    lon_max = reef_data['Longitude'].max() + 0.05
    lat_min = reef_data['Latitude'].min() - 0.08
    lat_max = reef_data['Latitude'].max() + 0.08
    
    # Subset data
    lon_mask = (lon >= lon_min) & (lon <= lon_max)
    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    
    lon_sub = lon[lon_mask]
    lat_sub = lat[lat_mask]
    u_sub = u_mean[np.ix_(lat_mask, lon_mask)]
    v_sub = v_mean[np.ix_(lat_mask, lon_mask)]
    mask_sub = mask[np.ix_(lat_mask, lon_mask)]
    
    # Calculate current speed
    speed = np.sqrt(u_sub**2 + v_sub**2)
    speed = np.where(mask_sub == 1, speed, np.nan)
    
    # Create figure
    fig = go.Figure()
    
    # Add current speed as background
    fig.add_trace(go.Heatmap(
        x=lon_sub,
        y=lat_sub,
        z=speed,
        colorscale='Blues',
        colorbar=dict(title="Speed<br>(m/s)"),
        hovertemplate='Speed: %{z:.3f} m/s<extra></extra>'
    ))
    
    # Add quiver plot (subsample for clarity)
    skip = 3
    lon_q = lon_sub[::skip]
    lat_q = lat_sub[::skip]
    u_q = u_sub[::skip, ::skip]
    v_q = v_sub[::skip, ::skip]
    
    # Create quiver as scatter plot with arrows
    for i, lat_val in enumerate(lat_q):
        for j, lon_val in enumerate(lon_q):
            if not np.isnan(u_q[i, j]) and not np.isnan(v_q[i, j]):
                # Scale arrows
                scale = 0.3
                fig.add_annotation(
                    x=lon_val,
                    y=lat_val,
                    ax=lon_val + u_q[i, j] * scale,
                    ay=lat_val + v_q[i, j] * scale,
                    xref="x",
                    yref="y",
                    axref="x",
                    ayref="y",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=1,
                    arrowcolor="black"
                )
    
    # Add reef locations
    fig.add_trace(go.Scatter(
        x=reef_data['Longitude'],
        y=reef_data['Latitude'],
        mode='markers',
        marker=dict(
            size=10,
            color='red',
            line=dict(color='white', width=1)
        ),
        text=reef_data['SourceReef'],
        hovertemplate='%{text}<extra></extra>',
        name='Reefs'
    ))
    
    fig.update_layout(
        title="Water Currents in St. Mary's River",
        xaxis_title="Longitude",
        yaxis_title="Latitude",
        height=600,
        template="plotly_white",
        xaxis=dict(scaleanchor="y", scaleratio=np.cos(np.radians(lat_sub.mean()))),
        showlegend=False
    )
    
    return fig, speed, mask_sub

def display_statistics(speed, mask_sub):
    """Display current statistics"""
    if speed is not None and mask_sub is not None:
        valid_speed = speed[mask_sub == 1]
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Current Speed", f"{np.nanmean(valid_speed):.3f} m/s")
        with col2:
            st.metric("Max Current Speed", f"{np.nanmax(valid_speed):.3f} m/s")
        with col3:
            st.metric("Dominant Direction", "Southwest")

def render_section(reef_metrics):
    """Render the complete water currents section"""
    st.header("Water Current Patterns")
    st.markdown("""
    Ocean currents drive larval transport. Arrows show current direction and speed is indicated by color intensity.
    """)
    
    result = create_visualization(reef_metrics)
    if result:
        fig, speed, mask_sub = result
        st.plotly_chart(fig, use_container_width=True)
        display_statistics(speed, mask_sub)
    else:
        st.warning("NetCDF data not available for current visualization")