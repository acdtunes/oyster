# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an R-based scientific analysis pipeline for modeling oyster larval dispersal patterns in the Chesapeake Bay region. The system uses oceanographic NetCDF data and reef coordinate data to model larval connectivity, settlement probabilities, and habitat suitability for Eastern oyster (Crassostrea virginica) populations.

## Architecture

The codebase follows a modular architecture with clear separation of concerns:

1. **Configuration Layer** (`R/config.R`): Centralized parameter management including file paths, model parameters, and analysis settings. All parameters are stored in a single `config` list object.

2. **Data Pipeline**: 
   - `R/data_loading.R`: Handles NetCDF (oceanographic) and Excel (reef location) data ingestion
   - Data flows through environmental analysis → dispersal modeling → statistical analysis → visualization

3. **Analysis Modules**:
   - Environmental processing calculates derived variables (current speed, stratification) and temporal statistics
   - Dispersal modeling uses biophysical parameters to build connectivity matrices
   - Network analysis classifies reefs as Sources, Sinks, or Hubs based on connectivity patterns

4. **Orchestration Scripts** (`scripts/`): High-level scripts that coordinate module execution for complete analyses

## Key Commands

### Running Analyses
```bash
# Full analysis pipeline (may take 30+ minutes for large datasets)
Rscript scripts/run_full_analysis.R

# St. Mary's River focused analysis (faster, ~2-5 minutes)
Rscript scripts/run_st_marys_analysis.R

# Custom analysis examples
Rscript scripts/example_usage.R
```

### Package Installation
```r
# Install required packages if missing
install.packages(c("ncdf4", "readxl", "dplyr", "tidyr", "ggplot2", "viridis", "RColorBrewer", "gridExtra"))

# Note: ncdf4 requires NetCDF C libraries on macOS:
# brew install netcdf
```

### Testing Individual Modules
```r
# In R console
setwd("/Users/andres.camacho/Development/ai/larval-oyster")
source("R/config.R")
source("R/data_loading.R")

# Test data loading
nc_data <- load_netcdf_data(config$paths$nc_file)
print(nc_data)  # Should show grid dimensions and variables
```

## Critical Implementation Details

### NetCDF Array Dimensions
The NetCDF data uses dimension ordering `[lon, lat, time]` not `[time, lat, lon]`. When extracting values at specific locations:
```r
# CORRECT
var_data[lon_idx, lat_idx, time_idx]

# INCORRECT (will cause errors)
var_data[time_idx, lat_idx, lon_idx]
```

### Connectivity Matrix Calculation
The dispersal model (`R/dispersal_modeling.R`) implements an advection-diffusion model with:
- Gaussian dispersal kernel modified by currents
- Daily mortality (default 10%)
- Pelagic larval duration of 21 days
- Settlement competency after 14 days

The connectivity calculation is computationally intensive (O(n²) for n reefs) and may require optimization for large reef networks.

### Data File Locations
Input data files are stored in `data/`:
- `109516.nc`: 114MB NetCDF with 49 environmental variables (348x567x12 grid)
- `Source_Reefs_Coordinates.xlsx`: Excel file with 30 reef locations

## Common Issues and Solutions

### Memory/Performance Issues
For large grids, use spatial subsetting:
```r
bounds <- list(lon_min=-76.5, lon_max=-76.3, lat_min=38.1, lat_max=38.3)
nc_data <- load_netcdf_data(config$paths$nc_file, subset_bounds=bounds)
```

### Variable Name Mismatches
NetCDF variables use specific naming:
- Use `temperature_surface` not `temp_surface`
- Use `u_surface`, `v_surface` for currents
- Check available variables: `names(nc$var)`

### Missing Print Methods
The custom S3 classes (`netcdf_data`, `reef_data`) have print methods defined in `data_loading.R`. Ensure this module is sourced before printing these objects.

## Model Parameters

Key biological parameters in `config$dispersal`:
- `pelagic_larval_duration`: 21 days (time larvae spend in water column)
- `mortality_rate`: 0.1 (10% daily mortality)
- `settlement_competency`: 14 days (when larvae can settle)
- `diffusion_coefficient`: 100 m²/s (horizontal dispersion)

Environmental optimal ranges in `config$optimal_ranges`:
- Temperature: 20-28°C
- Salinity: 14-28 PSU
- pH: 7.8-8.3
- Dissolved oxygen: 5-10 mg/L

## Output Structure

Analyses generate outputs in organized directories:
- `output/figures/`: Maps and plots (PNG format)
- `output/tables/`: CSV data exports
- `output/reports/`: Text analysis reports
- `output/models/`: Saved R model objects (RDS)
- `output/st_marys/`: St. Mary's River specific outputs

## Scientific Context

This pipeline models oyster larval dispersal using:
1. **Habitat Suitability Index (HSI)**: Combines multiple environmental variables to assess habitat quality
2. **Connectivity Matrices**: Quantifies larval exchange between reef pairs
3. **Network Metrics**: Identifies ecologically important reefs (sources, sinks, stepping stones)
4. **Settlement Fields**: Maps probability of larval settlement across the study area

The St. Mary's River analysis focuses on a specific region with 28 reef sites, showing high self-recruitment (43.7%) due to low current speeds (0.05 m/s).