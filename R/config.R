################################################################################
# Configuration File for Oyster Larval Dispersal Analysis
# All parameters and settings in one place
################################################################################

#' @title Configuration settings for larval dispersal analysis
#' @description Central configuration for all analysis parameters

# File paths
config <- list(
  # Data files
  paths = list(
    nc_file = "/Users/andres.camacho/Development/ai/larval-oyster/data/109516.nc",
    excel_file = "/Users/andres.camacho/Development/ai/larval-oyster/data/Source_Reefs_Coordinates.xlsx",
    output_dir = "output",
    figures_dir = "output/figures",
    tables_dir = "output/tables",
    reports_dir = "output/reports",
    models_dir = "output/models"
  ),
  
  # Dispersal model parameters
  dispersal = list(
    pelagic_larval_duration = 21,    # days
    mortality_rate = 0.1,             # daily mortality rate
    settlement_competency = 14,       # days until competent to settle
    max_dispersal_distance = 100,     # km
    diffusion_coefficient = 100,      # m²/s
    swimming_speed = 0.001,           # m/s (weak swimmers)
    vertical_migration = TRUE         # diel vertical migration
  ),
  
  # Environmental optimal ranges for Eastern oyster (Crassostrea virginica)
  optimal_ranges = list(
    temperature = c(20, 28),          # °C
    salinity = c(14, 28),             # PSU
    pH = c(7.8, 8.3),
    O2 = c(5, 10),                    # mg/L
    depth = c(-10, -0.5)              # meters (shallow subtidal)
  ),
  
  # Visualization settings
  visualization = list(
    color_schemes = list(
      temperature = "plasma",
      salinity = "viridis",
      current = "cividis",
      hsi = "mako",
      connectivity = c("white", "yellow", "orange", "red")
    ),
    figure_width = 10,
    figure_height = 8,
    dpi = 300,
    vector_scale = 0.3,               # Scale factor for current vectors
    vector_skip = 3                   # Subsample rate for vectors
  ),
  
  # Analysis parameters
  analysis = list(
    spawning_months = 5:9,            # May through September
    min_connectivity_threshold = 0.01, # Minimum connectivity to consider
    hsi_threshold = 0.7,              # Threshold for suitable habitat
    settlement_threshold = 0.5        # Threshold for high settlement
  ),
  
  # Report settings
  report = list(
    author = "R Analysis Pipeline v2.0",
    date_format = "%Y-%m-%d",
    include_figures = TRUE,
    include_tables = TRUE,
    verbose = TRUE
  )
)

#' @title Get configuration value
#' @param path Dot-separated path to config value (e.g., "paths.nc_file")
#' @return Configuration value
get_config <- function(path) {
  keys <- strsplit(path, "\\.")[[1]]
  value <- config
  for (key in keys) {
    value <- value[[key]]
  }
  return(value)
}

#' @title Update configuration value
#' @param path Dot-separated path to config value
#' @param value New value to set
set_config <- function(path, value) {
  keys <- strsplit(path, "\\.")[[1]]
  # This is simplified - in production would need recursive update
  if (length(keys) == 2) {
    config[[keys[1]]][[keys[2]]] <<- value
  }
}

#' @title Validate configuration
#' @return TRUE if valid, otherwise stops with error
validate_config <- function() {
  # Check if data files exist
  if (!file.exists(config$paths$nc_file)) {
    stop(paste("NetCDF file not found:", config$paths$nc_file))
  }
  if (!file.exists(config$paths$excel_file)) {
    stop(paste("Excel file not found:", config$paths$excel_file))
  }
  
  # Validate parameter ranges
  if (config$dispersal$mortality_rate < 0 || config$dispersal$mortality_rate > 1) {
    stop("Mortality rate must be between 0 and 1")
  }
  
  if (config$dispersal$pelagic_larval_duration <= 0) {
    stop("Pelagic larval duration must be positive")
  }
  
  return(TRUE)
}

# Export configuration
invisible(config)