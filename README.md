# Oyster Larval Dispersal Analysis Pipeline

A comprehensive R-based analysis system for modeling Eastern oyster (*Crassostrea virginica*) larval dispersal patterns using oceanographic data and reef locations in the Chesapeake Bay region.

## Overview

This pipeline integrates biophysical oceanographic data with oyster reef locations to model larval connectivity patterns, assess habitat suitability, and identify ecologically important reef sites for restoration and management. The system uses a modular architecture that allows for both complete bay-wide analyses and focused regional studies.

## Key Features

- **Biophysical Dispersal Modeling**: Advection-diffusion model with biological parameters
- **Connectivity Analysis**: Quantifies larval exchange between reef pairs
- **Habitat Suitability Assessment**: Multi-factor environmental suitability index
- **Network Classification**: Identifies source, sink, and hub reefs
- **Settlement Probability Mapping**: Spatial predictions of larval settlement
- **Automated Reporting**: Generates comprehensive analysis reports with visualizations

## Installation

### Prerequisites

- R (≥ 4.0.0)
- NetCDF C libraries (for macOS: `brew install netcdf`)

### R Package Dependencies

```r
# Install required packages
install.packages(c(
  "ncdf4",         # NetCDF file handling
  "readxl",        # Excel file reading
  "dplyr",         # Data manipulation
  "tidyr",         # Data reshaping
  "ggplot2",       # Visualization
  "viridis",       # Color scales
  "RColorBrewer",  # Color palettes
  "gridExtra"      # Multi-panel plots
))
```

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/larval-oyster.git
cd larval-oyster
```

2. Place your data files in the `data/` directory:
   - NetCDF file with oceanographic variables
   - Excel file with reef coordinates and density data

3. Update file paths in `R/config.R` if needed

## Quick Start

### Run Complete Analysis

```bash
# Full bay-wide analysis
Rscript scripts/run_full_analysis.R

# St. Mary's River focused analysis
Rscript scripts/run_st_marys_analysis.R
```

### Custom Analysis

```r
# Load modules
source("R/config.R")
source("R/data_loading.R")
source("R/dispersal_modeling.R")

# Load data
nc_data <- load_netcdf_data(config$paths$nc_file)
reef_data <- load_reef_data(config$paths$excel_file)

# Build connectivity matrix
conn_matrix <- build_connectivity_matrix(
  reef_data = reef_data,
  nc_data = nc_data,
  params = config$dispersal
)

# Calculate metrics
metrics <- calculate_network_metrics(conn_matrix, reef_data)
```

## Project Structure

```
larval-oyster/
├── R/                          # Core analysis modules
│   ├── config.R               # Configuration and parameters
│   ├── data_loading.R         # Data ingestion utilities
│   ├── environmental_analysis.R # Environmental processing
│   ├── dispersal_modeling.R   # Connectivity calculations
│   ├── visualization.R        # Plotting functions
│   ├── statistical_analysis.R # Statistical methods
│   └── report_generation.R    # Report creation
│
├── scripts/                    # Analysis orchestration
│   ├── run_full_analysis.R   # Complete pipeline
│   ├── run_st_marys_analysis.R # Regional analysis
│   └── example_usage.R       # Usage examples
│
├── data/                      # Input data files
│   ├── *.nc                  # NetCDF oceanographic data
│   └── *.xlsx                # Reef coordinates
│
└── output/                    # Generated results
    ├── figures/              # Maps and plots
    ├── tables/               # CSV exports
    ├── reports/              # Analysis reports
    └── models/               # Saved R objects
```

## Input Data Requirements

### NetCDF Oceanographic Data

Required variables:
- **Dimensions**: `longitude`, `latitude`, `time`
- **Currents**: `u_surface`, `v_surface` (m/s)
- **Water quality**: `temperature_surface` (°C), `salinity_surface` (PSU), `pH_surface`, `O2_surface` (mg/L)
- **Bathymetry**: `mask_land_sea`, `topography` (m)

### Reef Location Data (Excel)

Required columns:
- `SourceReef`: Unique reef identifier
- `Longitude`: Decimal degrees (negative for west)
- `Latitude`: Decimal degrees
- `AvgDensity`: Oyster density (individuals/m²)

## Model Parameters

### Biological Parameters

- **Pelagic Larval Duration**: 21 days
- **Daily Mortality Rate**: 10%
- **Settlement Competency**: 14 days
- **Swimming Speed**: 0.001 m/s
- **Diffusion Coefficient**: 100 m²/s

### Environmental Optima

- **Temperature**: 20-28°C
- **Salinity**: 14-28 PSU
- **pH**: 7.8-8.3
- **Dissolved Oxygen**: 5-10 mg/L

Parameters can be modified in `R/config.R`.

## Outputs

### Visualizations
- Environmental condition maps (temperature, salinity, currents)
- Habitat Suitability Index (HSI) maps
- Connectivity matrices with hierarchical clustering
- Settlement probability fields
- Current vector fields
- Network classification plots

### Data Products
- `reef_metrics.csv`: Network metrics for each reef (in/out strength, betweenness)
- `connectivity_matrix.csv`: Pairwise connectivity values
- `environmental_summary.csv`: Environmental statistics
- `monthly_statistics.csv`: Seasonal variations

### Reports
- Comprehensive analysis reports with key findings
- Management recommendations
- Statistical summaries

## Example Results

### St. Mary's River Analysis
- **Study Area**: 28 reef sites
- **Mean Connectivity**: 0.437
- **Self-recruitment**: 43.7%
- **Suitable Settlement Area**: 83.9% of water area
- **Key Finding**: Low current speeds (0.05 m/s) promote local larval retention

## Scientific Background

The model implements a biophysical dispersal framework that combines:

1. **Physical Transport**: Advection by currents and turbulent diffusion
2. **Biological Behavior**: Mortality, swimming, vertical migration
3. **Settlement Dynamics**: Competency period and habitat preferences
4. **Network Analysis**: Graph theory metrics to identify ecological roles

## Performance Considerations

- Full analysis on 348×567 grid may take 30+ minutes
- Regional analyses (e.g., St. Mary's) complete in 2-5 minutes
- For large datasets, use spatial subsetting:

```r
bounds <- list(lon_min=-76.5, lon_max=-76.3, lat_min=38.1, lat_max=38.3)
nc_data <- load_netcdf_data(config$paths$nc_file, subset_bounds=bounds)
```

