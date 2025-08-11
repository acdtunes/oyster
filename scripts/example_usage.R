#!/usr/bin/env Rscript
################################################################################
# Example Usage of Modular Functions
# Demonstrates how to use individual modules for custom analyses
################################################################################

# Set working directory
setwd("/Users/andres.camacho/Development/ai/larval-oyster")

# Load only the modules you need
source("R/config.R")
source("R/data_loading.R")
source("R/environmental_analysis.R")
source("R/visualization.R")

# Load packages
load_required_packages()
load_viz_packages()

cat("Example: Custom Environmental Analysis\n")
cat("======================================\n\n")

################################################################################
# EXAMPLE 1: Load and explore a specific region
################################################################################

cat("Example 1: Loading data for a specific region\n")
cat("----------------------------------------------\n")

# Define custom bounds (e.g., focus on a smaller area)
custom_bounds <- list(
  lon_min = -76.5,
  lon_max = -76.3,
  lat_min = 38.1,
  lat_max = 38.3
)

# Load only specific variables
nc_subset <- load_netcdf_data(
  nc_file = config$paths$nc_file,
  variables = c("temperature_surface", "salinity_surface", "mask_land_sea"),
  subset_bounds = custom_bounds
)

# Print summary
print(nc_subset)

################################################################################
# EXAMPLE 2: Calculate custom environmental statistics
################################################################################

cat("\nExample 2: Custom environmental analysis\n")
cat("-----------------------------------------\n")

# Calculate different statistics
temp_mean <- calculate_temporal_stats(nc_subset, "mean")
temp_max <- calculate_temporal_stats(nc_subset, "max")
temp_min <- calculate_temporal_stats(nc_subset, "min")

# Create temperature range map
temp_range <- temp_max$max_temperature_surface - temp_min$min_temperature_surface

# Visualize
p_range <- plot_environmental_map(
  env_data = temp_range,
  lon = nc_subset$dimensions$lon,
  lat = nc_subset$dimensions$lat,
  title = "Annual Temperature Range",
  var_name = "Range (°C)",
  color_scale = "plasma"
)

save_plot(p_range, "output/figures/temperature_range_example.png")
cat("✓ Temperature range map created\n")

################################################################################
# EXAMPLE 3: Filter and analyze specific reefs
################################################################################

cat("\nExample 3: Analyzing specific reef subset\n")
cat("------------------------------------------\n")

# Load all reefs
all_reefs <- load_reef_data(config$paths$excel_file)

# Filter for high-density reefs only
high_density_reefs <- all_reefs %>%
  filter(AvgDensity > 200)

cat(sprintf("Found %d high-density reefs (>200 oysters/m²)\n", 
            nrow(high_density_reefs)))

# Extract environmental conditions
reef_env <- extract_reef_environment(high_density_reefs, nc_subset)

# Summary statistics
cat("\nEnvironmental conditions at high-density reefs:\n")
cat(sprintf("  Temperature: %.1f ± %.1f°C\n",
            mean(reef_env$temperature, na.rm=TRUE),
            sd(reef_env$temperature, na.rm=TRUE)))
cat(sprintf("  Salinity: %.1f ± %.1f PSU\n",
            mean(reef_env$salinity, na.rm=TRUE),
            sd(reef_env$salinity, na.rm=TRUE)))

################################################################################
# EXAMPLE 4: Custom HSI with modified parameters
################################################################################

cat("\nExample 4: Custom Habitat Suitability Index\n")
cat("--------------------------------------------\n")

# Define stricter optimal ranges
strict_ranges <- list(
  temperature = c(22, 26),    # Narrower than default
  salinity = c(18, 25),       # Narrower than default
  pH = c(8.0, 8.2),          # Narrower than default
  O2 = c(6, 9)               # Narrower than default
)

# Calculate HSI with strict criteria
nc_full <- load_netcdf_data(config$paths$nc_file)
nc_full <- calculate_derived_variables(nc_full)
env_means <- calculate_temporal_stats(nc_full, "mean")

hsi_strict <- calculate_hsi(env_means, strict_ranges, 
                           nc_full$variables$mask_land_sea)

# Compare with default HSI
hsi_default <- calculate_hsi(env_means, config$optimal_ranges,
                            nc_full$variables$mask_land_sea)

# Calculate difference
suitable_default <- sum(hsi_default > 0.7 & nc_full$variables$mask_land_sea == 1)
suitable_strict <- sum(hsi_strict > 0.7 & nc_full$variables$mask_land_sea == 1)
total_marine <- sum(nc_full$variables$mask_land_sea == 1)

cat(sprintf("Suitable habitat (HSI > 0.7):\n"))
cat(sprintf("  Default ranges: %.1f%% of marine area\n", 
            suitable_default/total_marine * 100))
cat(sprintf("  Strict ranges:  %.1f%% of marine area\n",
            suitable_strict/total_marine * 100))

################################################################################
# EXAMPLE 5: Time series extraction
################################################################################

cat("\nExample 5: Extracting time series at reef locations\n")
cat("----------------------------------------------------\n")

# Get top 3 reefs by density
top_reefs <- head(all_reefs[order(all_reefs$AvgDensity, decreasing=TRUE), ], 3)

# Extract time series
time_series <- extract_time_series(
  nc_data = nc_full,
  locations = top_reefs,
  variables = c("temperature_surface", "salinity_surface")
)

# Plot temperature time series
library(ggplot2)
p_ts <- ggplot(time_series, aes(x = time, y = temperature_surface, 
                                color = location)) +
  geom_line(size = 1.2) +
  geom_point(size = 2) +
  scale_x_continuous(breaks = 1:12, labels = month.abb) +
  labs(title = "Temperature Time Series at Top Reef Sites",
       x = "Month", y = "Temperature (°C)",
       color = "Reef") +
  theme_dispersal()

save_plot(p_ts, "output/figures/temperature_timeseries_example.png")
cat("✓ Time series plot created\n")

################################################################################
# EXAMPLE 6: Batch processing multiple scenarios
################################################################################

cat("\nExample 6: Testing multiple dispersal scenarios\n")
cat("------------------------------------------------\n")

# Load dispersal module for this example
source("R/dispersal_modeling.R")

# Define scenarios
scenarios <- list(
  short_pld = list(pelagic_larval_duration = 14),
  standard_pld = list(pelagic_larval_duration = 21),
  long_pld = list(pelagic_larval_duration = 28)
)

# Run each scenario (simplified for example)
results <- list()

for (scenario_name in names(scenarios)) {
  cat(sprintf("  Running scenario: %s\n", scenario_name))
  
  # Update parameters
  params <- config$dispersal
  params$pelagic_larval_duration <- scenarios[[scenario_name]]$pelagic_larval_duration
  
  # Calculate connectivity (using subset for speed)
  sample_reefs <- head(all_reefs, 10)  # Use only 10 reefs for example
  
  conn_matrix <- build_connectivity_matrix(
    reef_data = sample_reefs,
    nc_data = nc_full,
    params = params,
    verbose = FALSE
  )
  
  # Store results
  results[[scenario_name]] <- list(
    mean_connectivity = mean(conn_matrix[conn_matrix > 0]),
    self_recruitment = mean(diag(conn_matrix))
  )
}

# Compare results
cat("\nScenario comparison:\n")
for (scenario_name in names(results)) {
  cat(sprintf("  %s: Mean connectivity = %.4f, Self-recruitment = %.4f\n",
              scenario_name,
              results[[scenario_name]]$mean_connectivity,
              results[[scenario_name]]$self_recruitment))
}

################################################################################
# SUMMARY
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("EXAMPLE ANALYSES COMPLETE\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("This script demonstrated:\n")
cat("  1. Loading data for specific regions\n")
cat("  2. Custom environmental statistics\n")
cat("  3. Filtering and analyzing reef subsets\n")
cat("  4. Modified HSI calculations\n")
cat("  5. Time series extraction\n")
cat("  6. Batch scenario processing\n\n")

cat("Outputs saved in: output/figures/\n")
cat("\nFeel free to modify and extend these examples for your analysis!\n")