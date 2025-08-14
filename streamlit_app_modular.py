#!/usr/bin/env python3
"""
St. Mary's River Oyster Larval Dispersal Analysis
Modular version with separated visualization components
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Import modules
from app_modules import data_loader
from app_modules import connectivity_matrix
from app_modules import distance_decay
from app_modules import settlement_map
from app_modules import water_currents
from app_modules import network_analysis

# Page configuration
st.set_page_config(
    page_title="Oyster Larval Dispersal Analysis",
    page_icon="🦪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #cccccc;
        padding: 10px;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

def create_sidebar(conn_matrix, reef_metrics, n_reefs):
    """Create sidebar with summary metrics"""
    with st.sidebar:
        st.header("📊 Summary Statistics")
        
        # Calculate key metrics
        conn_matrix_subset = conn_matrix[:n_reefs, :n_reefs]
        mean_connectivity = conn_matrix_subset[conn_matrix_subset > 0].mean()
        self_recruitment = conn_matrix_subset.diagonal().mean()
        max_connectivity = conn_matrix_subset.max()
        
        st.metric("Number of Reefs", n_reefs)
        st.metric("Mean Connectivity", f"{mean_connectivity:.4f}")
        st.metric("Mean Self-Recruitment", f"{self_recruitment:.3f}")
        st.metric("Max Connectivity", f"{max_connectivity:.3f}")
        
        st.markdown("---")
        st.markdown("### 📈 Analysis Controls")
        
        # Add any controls here if needed
        show_annotations = st.checkbox("Show annotations", value=True)
        color_scheme = st.selectbox("Color scheme", ["Viridis", "Blues", "YlOrRd", "RdBu"])
        
        return show_annotations, color_scheme

def main():
    """Main application"""
    # Header
    st.title("🦪 St. Mary's River Oyster Larval Dispersal Analysis")
    st.markdown("*Scientific visualization of connectivity patterns and dispersal dynamics*")
    
    # Load data
    conn_matrix, reef_metrics, n_reefs = data_loader.load_connectivity_data()
    
    if conn_matrix is None or reef_metrics is None:
        st.error("Please ensure data files are available in output/st_marys/")
        st.info("Required files:")
        st.code("""
        - output/st_marys/connectivity_matrix.csv
        - output/st_marys/reef_metrics.csv
        - data/109516.nc (optional, for currents)
        """)
        return
    
    # Create sidebar
    show_annotations, color_scheme = create_sidebar(conn_matrix, reef_metrics, n_reefs)
    
    # Main content with tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Connectivity Matrix", 
        "📉 Distance Decay", 
        "🗺️ Settlement Map",
        "🌊 Water Currents",
        "🕸️ Network Analysis"
    ])
    
    with tab1:
        connectivity_matrix.render_section(conn_matrix, reef_metrics, n_reefs)
    
    with tab2:
        distance_decay.render_section(conn_matrix, reef_metrics, n_reefs)
    
    with tab3:
        settlement_map.render_section(reef_metrics)
    
    with tab4:
        water_currents.render_section(reef_metrics)
    
    with tab5:
        network_analysis.render_section(conn_matrix, reef_metrics, n_reefs)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: gray; padding: 20px;'>
        Oyster Larval Dispersal Analysis | St. Mary's River, Maryland<br>
        Data from ROMS oceanographic model and field surveys
        </div>
        """, 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()