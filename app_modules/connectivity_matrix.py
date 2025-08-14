"""
Connectivity Matrix visualization module
"""
import numpy as np
import plotly.graph_objects as go
import streamlit as st

def create_visualization(conn_matrix, reef_metrics, n_reefs):
    """Create interactive connectivity matrix heatmap"""
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]
    reef_names = reef_metrics['SourceReef'].iloc[:n_reefs].values
    
    # Create hover text with reef names and values
    hover_text = []
    for i in range(n_reefs):
        row_text = []
        for j in range(n_reefs):
            row_text.append(f"From: {reef_names[i]}<br>To: {reef_names[j]}<br>Connectivity: {conn_matrix[i,j]:.4f}")
        hover_text.append(row_text)
    
    fig = go.Figure(data=go.Heatmap(
        z=conn_matrix,
        x=reef_names,
        y=reef_names,
        colorscale='Viridis',
        colorbar=dict(title="Connectivity<br>Strength"),
        hovertext=hover_text,
        hovertemplate='%{hovertext}<extra></extra>',
        zmin=0,
        zmax=np.percentile(conn_matrix[conn_matrix > 0], 95)
    ))
    
    fig.update_layout(
        title="Reef Connectivity Matrix",
        xaxis_title="Destination Reef",
        yaxis_title="Source Reef",
        height=700,
        xaxis=dict(tickangle=45, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
        template="plotly_white"
    )
    
    # Add diagonal line for self-recruitment
    fig.add_shape(
        type="line",
        x0=-0.5, y0=-0.5,
        x1=n_reefs-0.5, y1=n_reefs-0.5,
        line=dict(color="red", width=2, dash="dash"),
    )
    
    return fig

def display_statistics(conn_matrix, n_reefs):
    """Display connectivity statistics"""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Connections", np.sum(conn_matrix > 0))
    with col2:
        st.metric("Mean Self-Recruitment", f"{np.diag(conn_matrix).mean():.3f}")
    with col3:
        st.metric("Network Density", f"{np.sum(conn_matrix > 0) / (n_reefs**2):.3f}")

def render_section(conn_matrix, reef_metrics, n_reefs):
    """Render the complete connectivity matrix section"""
    st.header("Connectivity Matrix")
    st.markdown("""
    The connectivity matrix shows the strength of larval exchange between reef pairs. 
    Darker colors indicate stronger connections. The diagonal represents self-recruitment.
    """)
    
    fig = create_visualization(conn_matrix, reef_metrics, n_reefs)
    st.plotly_chart(fig, use_container_width=True)
    display_statistics(conn_matrix, n_reefs)