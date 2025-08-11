# 🦪 Oyster Larval Dispersal Analysis - Interactive Streamlit App

## Overview

This interactive Streamlit application provides a visually appealing exploration of oyster larval dispersal patterns in St. Mary's River, Chesapeake Bay. The app features animations, 3D visualizations, and interactive plots to help understand connectivity patterns and inform restoration decisions.

## Features

### 🎨 Visual Highlights
- **Animated Particle Tracking**: Watch 21-day larval dispersal simulations
- **3D Connectivity Matrix**: Interactive surface plot of reef connections
- **Network Analysis**: PageRank and centrality metrics visualization
- **Climate Projections**: Explore future scenarios (RCP 2.6 to 8.5)
- **Model Validation Gauges**: Visual validation metrics
- **Custom CSS Styling**: Beautiful gradients and animations

### 📊 Interactive Sections
1. **Overview**: Executive summary with key metrics
2. **Study Area**: Interactive map with reef locations
3. **Connectivity Matrix**: Static, 3D, and animated views
4. **Distance Decay**: Relationship between distance and connectivity
5. **Current Dynamics**: Animated particle tracking simulation
6. **Network Analysis**: Graph visualization with centrality metrics
7. **Model Validation**: Gauge charts showing model performance
8. **Future Directions**: Research priorities and climate projections

## Installation

### Prerequisites
- Python 3.9 or higher
- pip package manager

### Install Dependencies
```bash
python3 -m pip install -r requirements.txt
```

Or install packages individually:
```bash
python3 -m pip install streamlit plotly pandas numpy matplotlib seaborn Pillow scipy networkx scikit-learn
```

## Running the App

### Launch the Application
```bash
python3 -m streamlit run streamlit_app.py
```

The app will automatically open in your default browser at `http://localhost:8501`

### Alternative Launch Method
```bash
streamlit run streamlit_app.py
```

### Headless Mode (for servers)
```bash
python3 -m streamlit run streamlit_app.py --server.headless true --server.port 8501
```

## Using the App

### Navigation
- Use the **sidebar** to navigate between sections
- Adjust **animation speed** (1-30 fps) for smoother/faster animations
- Select different **color schemes** (Viridis, Plasma, Inferno, Ocean, Rainbow)

### Key Interactions

#### 🚀 Particle Tracking Simulation
1. Navigate to "Current Dynamics" section
2. Select number of particles (10-1000)
3. Choose release site (specific reef or all sites)
4. Click "Launch Simulation" to watch 21-day dispersal
5. Observe mortality effects as particles disappear

#### 🔗 Connectivity Matrix Animation
1. Go to "Connectivity Matrix" section
2. Select "Animated Build-up" tab
3. Click "Play Animation" to see connections form
4. Watch strongest connections appear first

#### 🎯 Network Analysis
1. Visit "Network Analysis" section
2. Explore interactive network graph
3. Hover over nodes to see PageRank scores
4. Review centrality metrics table

#### 🌡️ Climate Projections
1. Navigate to "Future Directions"
2. Select "Climate" tab
3. Choose RCP scenario (2.6, 4.5, 6.0, 8.5)
4. Adjust projection year (2030-2100)
5. View projected impacts on connectivity

### Optimization Tools
- **Restoration Site Optimizer**: Set budget and objectives
- **Research Priority Matrix**: Explore impact vs. feasibility
- **Management Applications**: Interactive planning tools

## Data Requirements

The app expects the following data files in the project directory:

### Required Files
- `output/st_marys/connectivity_matrix.csv` - Reef connectivity data
- `output/st_marys/reef_metrics.csv` - Reef characteristics

### Generated Visualizations
The app uses pre-generated figures from:
- `output/article_figures/` - Publication-quality figures
- `output/st_marys/` - St. Mary's River specific outputs

## Performance Tips

1. **Large Datasets**: The app caches data loading for faster performance
2. **Animations**: Reduce particle count for smoother animations on slower systems
3. **3D Plots**: May require more resources - close other applications if needed
4. **Browser**: Chrome or Firefox recommended for best performance

## Troubleshooting

### App Won't Start
```bash
# Check Python version
python3 --version

# Ensure all dependencies are installed
python3 -m pip list | grep streamlit

# Try explicit module path
python3 -m streamlit run ./streamlit_app.py
```

### Data Loading Errors
```bash
# Verify data files exist
ls -la output/st_marys/
```

### Port Already in Use
```bash
# Use a different port
python3 -m streamlit run streamlit_app.py --server.port 8502
```

### Clear Cache
```bash
# If data appears outdated
python3 -m streamlit cache clear
```

## Features Showcase

### Animated Header
- Pulsing title animation
- Gradient background
- Responsive design

### Interactive Controls
- Animation speed slider (1-30 fps)
- Color scheme selector
- Quick stats in sidebar
- Navigation radio buttons

### Visualizations
- **Plotly** for interactive plots
- **3D surfaces** for connectivity matrix
- **Mapbox** for geographic visualization
- **Network graphs** with spring layout
- **Gauge charts** for validation metrics
- **Radar charts** for multi-metric comparison

### Climate Scenarios
- RCP 2.6 (Low emissions)
- RCP 4.5 (Moderate)
- RCP 6.0 (High)
- RCP 8.5 (Very high)

## Customization

### Modify Color Schemes
Edit line 154 in `streamlit_app.py`:
```python
color_scheme = st.selectbox(
    "Color Scheme",
    ["Viridis", "Plasma", "Inferno", "Ocean", "Rainbow", "YourCustomScheme"]
)
```

### Adjust Animation Speed
Edit line 143:
```python
animation_speed = st.slider(
    "Animation Speed (fps)",
    min_value=1,
    max_value=60,  # Increase for faster animations
    value=10
)
```

### Add New Sections
Add to navigation in line 126:
```python
page = st.radio(
    "Select Section",
    ["🏠 Overview", 
     # ... existing sections ...
     "🆕 Your New Section"]
)
```

## Citation

If you use this application in your research, please cite:
```
Oyster Larval Dispersal Analysis - Interactive Streamlit Application
St. Mary's River, Chesapeake Bay
November 2024
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the data requirements
3. Ensure all dependencies are installed
4. Try running with `--log_level debug` for more information

## License

This application is part of the Oyster Larval Dispersal Analysis project for St. Mary's River restoration planning.

---

**App is currently running at:** http://localhost:8501

Enjoy exploring the fascinating world of oyster larval dispersal! 🦪🌊