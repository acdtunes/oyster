"""
Network Analysis visualization module
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def classify_reefs(out_strength, in_strength):
    """Classify reefs based on network metrics"""
    n_reefs = len(out_strength)
    reef_types = []
    
    for i in range(n_reefs):
        if out_strength[i] > np.percentile(out_strength, 75):
            if in_strength[i] > np.percentile(in_strength, 75):
                reef_types.append("Hub")
            else:
                reef_types.append("Source")
        elif in_strength[i] > np.percentile(in_strength, 75):
            reef_types.append("Sink")
        else:
            reef_types.append("Regular")
    
    return reef_types

def create_visualization(conn_matrix, reef_metrics, n_reefs):
    """Create network analysis visualization"""
    conn_matrix = conn_matrix[:n_reefs, :n_reefs]
    reef_data = reef_metrics.iloc[:n_reefs]
    
    # Calculate network metrics
    out_strength = conn_matrix.sum(axis=1)
    in_strength = conn_matrix.sum(axis=0)
    betweenness = (out_strength + in_strength) / 2
    self_recruitment = np.diag(conn_matrix)
    
    # Classify reefs
    reef_types = classify_reefs(out_strength, in_strength)
    
    # Create subplot figure
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Reef Classification", "Self-Recruitment", 
                       "Network Centrality", "Source-Sink Dynamics"),
        specs=[[{"type": "scatter"}, {"type": "bar"}],
               [{"type": "scatter"}, {"type": "scatter"}]]
    )
    
    # 1. Reef Classification Map
    color_map = {"Source": "red", "Sink": "blue", "Hub": "green", "Regular": "gray"}
    colors = [color_map[t] for t in reef_types]
    
    fig.add_trace(
        go.Scatter(
            x=reef_data['Longitude'],
            y=reef_data['Latitude'],
            mode='markers+text',
            marker=dict(
                size=15,
                color=colors,
                line=dict(color='white', width=2)
            ),
            text=[f"{name}<br>({type})" for name, type in zip(reef_data['SourceReef'], reef_types)],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='%{text}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # 2. Self-Recruitment Bar Chart
    fig.add_trace(
        go.Bar(
            x=reef_data['SourceReef'],
            y=self_recruitment,
            marker_color='teal',
            hovertemplate='%{x}<br>Self-recruitment: %{y:.3f}<extra></extra>'
        ),
        row=1, col=2
    )
    
    # 3. Network Centrality
    fig.add_trace(
        go.Scatter(
            x=out_strength,
            y=in_strength,
            mode='markers+text',
            marker=dict(
                size=betweenness * 100,
                color=betweenness,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Centrality", x=0.45, y=0.15)
            ),
            text=reef_data['SourceReef'],
            textposition="top center",
            textfont=dict(size=8),
            hovertemplate='%{text}<br>Out: %{x:.3f}<br>In: %{y:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # 4. Source-Sink Dynamics
    source_sink_balance = out_strength - in_strength
    fig.add_trace(
        go.Scatter(
            x=reef_data['Longitude'],
            y=reef_data['Latitude'],
            mode='markers',
            marker=dict(
                size=15,
                color=source_sink_balance,
                colorscale='RdBu',
                cmid=0,
                showscale=True,
                colorbar=dict(title="Balance", x=1.02, y=0.15)
            ),
            text=reef_data['SourceReef'],
            hovertemplate='%{text}<br>Balance: %{marker.color:.3f}<extra></extra>',
            showlegend=False
        ),
        row=2, col=2
    )
    
    # Update layout
    fig.update_xaxes(title_text="Longitude", row=1, col=1)
    fig.update_yaxes(title_text="Latitude", row=1, col=1)
    fig.update_xaxes(tickangle=45, row=1, col=2)
    fig.update_yaxes(title_text="Self-Recruitment", row=1, col=2)
    fig.update_xaxes(title_text="Out-Strength", row=2, col=1)
    fig.update_yaxes(title_text="In-Strength", row=2, col=1)
    fig.update_xaxes(title_text="Longitude", row=2, col=2)
    fig.update_yaxes(title_text="Latitude", row=2, col=2)
    
    fig.update_layout(
        height=800,
        template="plotly_white",
        showlegend=False
    )
    
    return fig, reef_types, out_strength, in_strength, self_recruitment

def display_summary_tables(reef_metrics, reef_types, out_strength, in_strength, self_recruitment):
    """Display reef classification summary tables"""
    st.subheader("Reef Classification Summary")
    
    n_reefs = len(reef_types)
    summary_df = pd.DataFrame({
        'Reef': reef_metrics['SourceReef'].iloc[:n_reefs],
        'Type': reef_types,
        'Out-Strength': out_strength,
        'In-Strength': in_strength,
        'Self-Recruitment': self_recruitment
    })
    summary_df = summary_df.sort_values('Out-Strength', ascending=False)
    
    # Display top reefs
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Top Source Reefs**")
        st.dataframe(summary_df.head(5)[['Reef', 'Out-Strength']], hide_index=True)
    with col2:
        st.markdown("**Top Sink Reefs**")
        sink_df = summary_df.sort_values('In-Strength', ascending=False)
        st.dataframe(sink_df.head(5)[['Reef', 'In-Strength']], hide_index=True)

def render_section(conn_matrix, reef_metrics, n_reefs):
    """Render the complete network analysis section"""
    st.header("Network Analysis")
    st.markdown("""
    Network analysis reveals the ecological roles of different reefs:
    - **Sources**: Export larvae to other reefs (red)
    - **Sinks**: Receive larvae from other reefs (blue)
    - **Hubs**: Both export and import larvae (green)
    - **Regular**: Average connectivity (gray)
    """)
    
    fig, reef_types, out_strength, in_strength, self_recruitment = create_visualization(
        conn_matrix, reef_metrics, n_reefs
    )
    st.plotly_chart(fig, use_container_width=True)
    
    display_summary_tables(reef_metrics, reef_types, out_strength, in_strength, self_recruitment)