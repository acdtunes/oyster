#!/usr/bin/env Rscript
################################################################################
# Main Analysis Script - Full Oyster Larval Dispersal Analysis
# This script orchestrates the complete analysis pipeline
################################################################################

# Clear workspace and set working directory
rm(list = ls())
setwd("/Users/andres.camacho/Development/ai/larval-oyster")

# Source all modules
cat("Loading analysis modules...\n")
source("R/config.R")
source("R/data_loading.R")
source("R/environmental_analysis.R")
source("R/dispersal_modeling.R")
source("R/visualization.R")
source("R/statistical_analysis.R")
source("R/report_generation.R")

# Load required packages
cat("Loading required packages...\n")
load_required_packages()
load_viz_packages()

# Validate configuration
validate_config()

################################################################################
# 1. DATA LOADING
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 1: LOADING DATA\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Load NetCDF data
nc_data <- load_netcdf_data(
  nc_file = config$paths$nc_file,
  variables = NULL,  # Load all variables
  subset_bounds = NULL  # Use full extent
)
print(nc_data)

# Load reef data
reef_data <- load_reef_data(
  excel_file = config$paths$excel_file,
  filter_prefix = NULL  # Load all reefs
)
print(reef_data)

# Extract environmental conditions at reefs
reef_env <- extract_reef_environment(reef_data, nc_data)

################################################################################
# 2. ENVIRONMENTAL ANALYSIS
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 2: ENVIRONMENTAL ANALYSIS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Calculate derived variables
nc_data <- calculate_derived_variables(nc_data)

# Calculate temporal statistics
env_means <- calculate_temporal_stats(nc_data, "mean")
env_sd <- calculate_temporal_stats(nc_data, "sd")

# Analyze seasonal patterns
seasonal <- analyze_seasonal_patterns(nc_data, config$analysis$spawning_months)

# Calculate HSI
hsi <- calculate_hsi(env_means, config$optimal_ranges, nc_data$variables$mask_land_sea)

# Environmental summary
env_summary <- summarize_environment(env_means, nc_data$variables$mask_land_sea)
cat("Environmental summary:\n")
print(env_summary)

################################################################################
# 3. DISPERSAL MODELING
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 3: DISPERSAL MODELING\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Build connectivity matrix
conn_matrix <- build_connectivity_matrix(
  reef_data = reef_data,
  nc_data = nc_data,
  params = config$dispersal,
  verbose = TRUE
)

# Calculate network metrics
metrics <- calculate_network_metrics(conn_matrix, reef_data)

# Add environmental data to metrics
metrics <- cbind(metrics, reef_env[, c("temperature", "salinity", "pH", "O2")])

# Calculate settlement field
settlement_field <- calculate_settlement_field(
  source_reefs = reef_data,
  nc_data = nc_data,
  params = config$dispersal
)

# Assess connectivity
conn_assessment <- assess_connectivity(conn_matrix)
cat("\nConnectivity assessment:\n")
cat(sprintf("  Mean connectivity: %.4f\n", conn_assessment$mean_connectivity))
cat(sprintf("  Strong connections: %d\n", conn_assessment$n_strong))
cat(sprintf("  Self-recruitment: %.3f\n", conn_assessment$mean_self_recruitment))

################################################################################
# 4. STATISTICAL ANALYSIS
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 4: STATISTICAL ANALYSIS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Correlation analysis
cor_results <- correlation_analysis(
  data = metrics,
  variables = c("Density", "temperature", "salinity", "OutStrength", "InStrength")
)

# Fit connectivity model
conn_model <- fit_connectivity_model(
  metrics = metrics,
  predictors = c("Density", "temperature", "salinity"),
  response = "OutStrength"
)
cat("Connectivity model summary:\n")
print(summary(conn_model))

# Calculate diversity indices
diversity <- calculate_diversity_indices(conn_matrix)
cat(sprintf("\nNetwork diversity - Shannon: %.3f, Simpson: %.3f\n", 
            diversity$shannon, diversity$simpson))

################################################################################
# 5. VISUALIZATION
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 5: CREATING VISUALIZATIONS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Temperature map
p1 <- plot_environmental_map(
  env_data = env_means$mean_temperature_surface,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = reef_data,
  title = "Mean Sea Surface Temperature",
  var_name = "Temperature (°C)",
  color_scale = "plasma"
)
save_plot(p1, file.path(config$paths$figures_dir, "temperature_map.png"))

# Salinity map
p2 <- plot_environmental_map(
  env_data = env_means$mean_salinity_surface,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = reef_data,
  title = "Mean Sea Surface Salinity",
  var_name = "Salinity (PSU)",
  color_scale = "viridis"
)
save_plot(p2, file.path(config$paths$figures_dir, "salinity_map.png"))

# Current vectors
p3 <- plot_current_vectors(
  u_field = env_means$mean_u_surface,
  v_field = env_means$mean_v_surface,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  mask = nc_data$variables$mask_land_sea,
  reef_data = reef_data,
  skip = config$visualization$vector_skip,
  scale_factor = config$visualization$vector_scale,
  title = "Mean Surface Currents"
)
save_plot(p3, file.path(config$paths$figures_dir, "current_vectors.png"))

# HSI map
p4 <- plot_environmental_map(
  env_data = hsi,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = reef_data,
  title = "Habitat Suitability Index",
  var_name = "HSI",
  color_scale = "mako",
  limits = c(0, 1)
)
save_plot(p4, file.path(config$paths$figures_dir, "hsi_map.png"))

# Connectivity matrix
p5 <- plot_connectivity_matrix(
  conn_matrix = conn_matrix,
  cluster = TRUE,
  title = "Larval Connectivity Matrix"
)
save_plot(p5, file.path(config$paths$figures_dir, "connectivity_matrix.png"),
         width = 12, height = 10)

# Network classification
p6 <- plot_network_classification(
  metrics = metrics,
  title = "Reef Network Classification"
)
save_plot(p6, file.path(config$paths$figures_dir, "network_classification.png"))

# Settlement probability
p7 <- plot_settlement_probability(
  settlement_field = settlement_field,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = reef_data,
  title = "Larval Settlement Probability"
)
save_plot(p7, file.path(config$paths$figures_dir, "settlement_probability.png"))

# Temporal variations
monthly_data <- data.frame(
  Month = 1:12,
  Temperature = seasonal$monthly$temperature_surface,
  Salinity = seasonal$monthly$salinity_surface,
  pH = seasonal$monthly$pH_surface
)
p8 <- plot_temporal_variation(
  monthly_data = monthly_data,
  variables = c("Temperature", "Salinity", "pH"),
  title = "Monthly Environmental Variations"
)
save_plot(p8, file.path(config$paths$figures_dir, "temporal_variations.png"),
         width = 12, height = 8)

cat("All visualizations created successfully\n")

################################################################################
# 6. SAVE RESULTS
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 6: SAVING RESULTS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Save data tables
export_summary_table(
  data = metrics,
  output_file = file.path(config$paths$tables_dir, "reef_metrics.csv")
)

export_summary_table(
  data = as.data.frame(conn_matrix),
  output_file = file.path(config$paths$tables_dir, "connectivity_matrix.csv")
)

export_summary_table(
  data = env_summary,
  output_file = file.path(config$paths$tables_dir, "environmental_summary.csv")
)

export_summary_table(
  data = monthly_data,
  output_file = file.path(config$paths$tables_dir, "monthly_statistics.csv")
)

# Save model objects
saveRDS(conn_model, file.path(config$paths$models_dir, "connectivity_model.rds"))
saveRDS(config$dispersal, file.path(config$paths$models_dir, "dispersal_parameters.rds"))

################################################################################
# 7. GENERATE REPORT
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("STEP 7: GENERATING REPORT\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

# Compile results
analysis_results <- list(
  reef_data = reef_data,
  env_summary = env_summary,
  monthly_data = monthly_data,
  conn_assessment = conn_assessment,
  metrics = metrics,
  config = config,
  hsi_stats = list(
    high_suitability_percent = sum(hsi > 0.8 & nc_data$variables$mask_land_sea == 1) /
                              sum(nc_data$variables$mask_land_sea == 1) * 100
  )
)

# Create report
create_full_report(
  analysis_results = analysis_results,
  output_file = file.path(config$paths$reports_dir, "full_analysis_report.txt"),
  include_plots = TRUE
)

################################################################################
# COMPLETION
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Summary of outputs:\n")
cat(sprintf("  ✓ %d figures generated\n", 8))
cat(sprintf("  ✓ %d data tables saved\n", 4))
cat(sprintf("  ✓ %d models saved\n", 2))
cat("  ✓ Comprehensive report generated\n\n")

cat("All results saved in the 'output' directory\n")
cat("Main report: output/reports/full_analysis_report.txt\n")