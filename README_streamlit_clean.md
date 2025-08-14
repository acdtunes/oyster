# Clean Streamlit App - Oyster Larval Dispersal Analysis

## Overview
This is a streamlined, scientific visualization tool for analyzing oyster larval dispersal patterns in St. Mary's River, Maryland. The app focuses on core analytical visualizations without unnecessary complexity.

## Features

### 1. 🔗 Connectivity Matrix
- Interactive heatmap showing larval exchange strength between reef pairs
- Diagonal elements represent self-recruitment rates
- Hover for detailed connectivity values

### 2. 📉 Distance Decay
- Scatter plot showing how connectivity decreases with distance
- Exponential decay model fitting
- Key metrics: correlation coefficient and 50% decay distance

### 3. 🗺️ Settlement Map
- Spatial visualization of larval settlement probability
- Gaussian kernel-based dispersal modeling
- Reef locations overlaid on probability field

### 4. 🌊 Water Currents
- Ocean current patterns from NetCDF data
- Current speed heatmap with directional arrows
- Reef locations in context of flow patterns

### 5. 🕸️ Network Analysis
- Reef classification (Source, Sink, Hub, Regular)
- Self-recruitment patterns
- Network centrality metrics
- Source-sink dynamics visualization

## Running the App

```bash
# Make sure you're in the project directory
cd /Users/andres.camacho/Development/ai/larval-oyster

# Run the clean app (uses port 8503)
python3 -m streamlit run streamlit_app_clean.py
```

## Data Requirements
The app expects the following data files:
- `output/st_marys/connectivity_matrix.csv` - Reef connectivity matrix
- `output/st_marys/reef_metrics.csv` - Reef characteristics and metrics
- `data/109516.nc` - NetCDF oceanographic data (optional for current visualization)

## Key Metrics Displayed
- **Number of Reefs**: Total reef sites analyzed
- **Mean Connectivity**: Average strength of connections
- **Mean Self-Recruitment**: Average local retention
- **Network Density**: Proportion of possible connections realized
- **Current Speed**: Ocean current velocities in m/s
- **Decay Distance**: Distance at which connectivity drops by 50%

## Sidebar Controls
- Toggle annotations on/off
- Select different color schemes
- View summary statistics

## Scientific Basis
The analysis implements:
- Advection-diffusion larval dispersal models
- Distance-decay relationships
- Network theory for ecological connectivity
- Gaussian dispersal kernels weighted by reef density

## Comparison with Original App
This clean version:
- Focuses on core scientific visualizations
- Removes complex animations and decorative elements
- Provides cleaner, more professional interface
- Optimizes performance with efficient data handling
- Uses consistent styling throughout