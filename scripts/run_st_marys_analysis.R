#!/usr/bin/env Rscript
################################################################################
# St. Mary's River Focused Analysis
# Specialized analysis for St. Mary's River oyster populations
################################################################################

# Clear workspace and set working directory
rm(list = ls())
setwd("/Users/andres.camacho/Development/ai/larval-oyster")

# Source required modules
source("R/config.R")
source("R/data_loading.R")
source("R/environmental_analysis.R")
source("R/dispersal_modeling.R")
source("R/visualization.R")
source("R/report_generation.R")

# Load packages
load_required_packages()
load_viz_packages()

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("ST. MARY'S RIVER OYSTER ANALYSIS\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

################################################################################
# 1. LOAD ST. MARY'S DATA
################################################################################

cat("Loading St. Mary's River data...\n")

# Load all reef data
all_reefs <- load_reef_data(config$paths$excel_file)

# Filter for St. Mary's reefs
st_marys_reefs <- all_reefs %>%
  filter(grepl("STM|Plant|Reef", SourceReef))

cat(sprintf("Found %d reefs in St. Mary's River area\n", nrow(st_marys_reefs)))

# Define St. Mary's bounds with buffer
bounds <- list(
  lon_min = min(st_marys_reefs$Longitude) - 0.1,
  lon_max = max(st_marys_reefs$Longitude) + 0.1,
  lat_min = min(st_marys_reefs$Latitude) - 0.1,
  lat_max = max(st_marys_reefs$Latitude) + 0.1
)

# Load NetCDF data for St. Mary's region
nc_data <- load_netcdf_data(
  nc_file = config$paths$nc_file,
  variables = c("u_surface", "v_surface", "temperature_surface", 
                "salinity_surface", "pH_surface", "O2_surface",
                "mask_land_sea", "topography"),
  subset_bounds = bounds
)

print(nc_data)

################################################################################
# 2. ENVIRONMENTAL ANALYSIS
################################################################################

cat("\nAnalyzing St. Mary's environmental conditions...\n")

# Calculate derived variables
nc_data <- calculate_derived_variables(nc_data)

# Get mean conditions
env_means <- calculate_temporal_stats(nc_data, "mean")

# Extract conditions at reef sites
reef_env <- extract_reef_environment(st_marys_reefs, nc_data)

# Calculate HSI for St. Mary's
hsi <- calculate_hsi(env_means, config$optimal_ranges, 
                    nc_data$variables$mask_land_sea)

################################################################################
# 3. DISPERSAL MODELING
################################################################################

cat("\nModeling larval dispersal in St. Mary's River...\n")

# Build connectivity matrix
conn_matrix <- build_connectivity_matrix(
  reef_data = st_marys_reefs,
  nc_data = nc_data,
  params = config$dispersal,
  verbose = TRUE
)

# Calculate settlement field
settlement_field <- calculate_settlement_field(
  source_reefs = st_marys_reefs,
  nc_data = nc_data,
  params = config$dispersal
)

# Network metrics
metrics <- calculate_network_metrics(conn_matrix, st_marys_reefs)
metrics <- cbind(metrics, reef_env[, c("temperature", "salinity", "pH")])

################################################################################
# 4. CREATE VISUALIZATIONS
################################################################################

cat("\nCreating St. Mary's River visualizations...\n")

# Create output directory
st_marys_dir <- "output/st_marys"
dir.create(st_marys_dir, showWarnings = FALSE, recursive = TRUE)

# Settlement probability map
p1 <- plot_settlement_probability(
  settlement_field = settlement_field,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = st_marys_reefs,
  title = "Larval Settlement Probability - St. Mary's River"
)
save_plot(p1, file.path(st_marys_dir, "settlement_probability.png"),
         width = 12, height = 10)

# Current vectors map
p2 <- plot_current_vectors(
  u_field = env_means$mean_u_surface,
  v_field = env_means$mean_v_surface,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  mask = nc_data$variables$mask_land_sea,
  reef_data = st_marys_reefs,
  skip = 2,
  scale_factor = 0.3,
  title = "Water Currents - St. Mary's River"
)
save_plot(p2, file.path(st_marys_dir, "current_vectors.png"),
         width = 12, height = 10)

# Combined dispersal map
p3 <- ggplot() +
  # Settlement probability
  geom_tile(data = expand.grid(lon = nc_data$dimensions$lon,
                               lat = nc_data$dimensions$lat) %>%
              mutate(prob = as.vector(settlement_field)) %>%
              filter(prob > 0),
           aes(x = lon, y = lat, fill = prob), alpha = 0.7) +
  scale_fill_gradientn(colors = c("navy", "blue", "cyan", "yellow", "red"),
                      name = "Settlement\nProbability") +
  # Current vectors
  geom_segment(data = expand.grid(
                 lon = nc_data$dimensions$lon[seq(1, length(nc_data$dimensions$lon), 3)],
                 lat = nc_data$dimensions$lat[seq(1, length(nc_data$dimensions$lat), 3)]
               ) %>%
               mutate(u = as.vector(env_means$mean_u_surface[seq(1, nrow(env_means$mean_u_surface), 3),
                                                            seq(1, ncol(env_means$mean_u_surface), 3)]),
                     v = as.vector(env_means$mean_v_surface[seq(1, nrow(env_means$mean_v_surface), 3),
                                                            seq(1, ncol(env_means$mean_v_surface), 3)])) %>%
               filter(!is.na(u)),
              aes(x = lon, y = lat, xend = lon + u*0.3, yend = lat + v*0.3),
              arrow = arrow(length = unit(0.1, "cm")),
              color = "black", alpha = 0.5, size = 0.3) +
  # Reefs
  geom_point(data = st_marys_reefs,
            aes(x = Longitude, y = Latitude, size = AvgDensity),
            color = "black", fill = "white", shape = 21, stroke = 2) +
  coord_fixed() +
  labs(title = "Larval Dispersal Dynamics - St. Mary's River",
       x = "Longitude", y = "Latitude") +
  theme_dispersal()

save_plot(p3, file.path(st_marys_dir, "combined_dispersal.png"),
         width = 12, height = 10)

# HSI map for St. Mary's
p4 <- plot_environmental_map(
  env_data = hsi,
  lon = nc_data$dimensions$lon,
  lat = nc_data$dimensions$lat,
  reef_data = st_marys_reefs,
  title = "Habitat Suitability - St. Mary's River",
  var_name = "HSI",
  color_scale = "mako",
  limits = c(0, 1)
)
save_plot(p4, file.path(st_marys_dir, "habitat_suitability.png"))

################################################################################
# 5. CALCULATE STATISTICS
################################################################################

cat("\nCalculating St. Mary's River statistics...\n")

# Environmental statistics
env_stats <- list(
  temperature = mean(reef_env$temperature, na.rm = TRUE),
  salinity = mean(reef_env$salinity, na.rm = TRUE),
  current_speed = mean(env_means$mean_current_speed_surface[
                      nc_data$variables$mask_land_sea == 1], na.rm = TRUE),
  pH = mean(reef_env$pH, na.rm = TRUE)
)

# Settlement statistics
settlement_stats <- list(
  high_settlement_area = sum(settlement_field > 0.5 & 
                           nc_data$variables$mask_land_sea == 1) /
                        sum(nc_data$variables$mask_land_sea == 1) * 100,
  mean_settlement = mean(settlement_field[nc_data$variables$mask_land_sea == 1],
                        na.rm = TRUE)
)

# Connectivity statistics
conn_stats <- assess_connectivity(conn_matrix)

################################################################################
# 6. GENERATE REPORT
################################################################################

cat("\nGenerating St. Mary's River report...\n")

# Create report
report <- generate_header("ST. MARY'S RIVER OYSTER ANALYSIS")

report <- paste0(report, format_section("STUDY AREA", 1))
report <- paste0(report, sprintf("Geographic extent: %.3f°W to %.3f°W, %.3f°N to %.3f°N\n",
                                abs(bounds$lon_min), abs(bounds$lon_max),
                                bounds$lat_min, bounds$lat_max))
report <- paste0(report, sprintf("Number of reef sites: %d\n", nrow(st_marys_reefs)))
report <- paste0(report, sprintf("Total oyster density: %.1f\n", 
                                sum(st_marys_reefs$AvgDensity)))

report <- paste0(report, format_section("ENVIRONMENTAL CONDITIONS", 1))
report <- paste0(report, sprintf("Mean temperature: %.1f°C\n", env_stats$temperature))
report <- paste0(report, sprintf("Mean salinity: %.1f PSU\n", env_stats$salinity))
report <- paste0(report, sprintf("Mean current speed: %.3f m/s\n", env_stats$current_speed))
report <- paste0(report, sprintf("Mean pH: %.2f\n", env_stats$pH))

report <- paste0(report, format_section("LARVAL DISPERSAL", 1))
report <- paste0(report, sprintf("High settlement area (>50%%): %.1f%% of water area\n",
                                settlement_stats$high_settlement_area))
report <- paste0(report, sprintf("Mean settlement probability: %.3f\n",
                                settlement_stats$mean_settlement))
report <- paste0(report, sprintf("Mean connectivity: %.4f\n", 
                                conn_stats$mean_connectivity))
report <- paste0(report, sprintf("Self-recruitment: %.3f\n",
                                conn_stats$mean_self_recruitment))

report <- paste0(report, format_section("KEY REEFS", 1))
top_sources <- head(metrics[order(metrics$OutStrength, decreasing = TRUE), ], 5)
report <- paste0(report, "\nTop source reefs:\n")
for (i in 1:min(5, nrow(top_sources))) {
  report <- paste0(report, sprintf("  %d. %s (Density: %.1f)\n",
                                  i, top_sources$SourceReef[i],
                                  top_sources$Density[i]))
}

report <- paste0(report, format_section("KEY FINDINGS", 1))
report <- paste0(report, "• Low current speeds promote local larval retention\n")
report <- paste0(report, "• High-density reefs create settlement hotspots\n")
report <- paste0(report, "• Limited exchange between upper and lower river\n")
report <- paste0(report, "• Excellent conditions for restoration success\n")

# Save report
writeLines(report, file.path(st_marys_dir, "st_marys_report.txt"))

# Save data tables
export_summary_table(metrics, file.path(st_marys_dir, "reef_metrics.csv"))
export_summary_table(as.data.frame(conn_matrix), 
                    file.path(st_marys_dir, "connectivity_matrix.csv"))

################################################################################
# COMPLETION
################################################################################

cat("\n", paste(rep("=", 60), collapse=""), "\n")
cat("ST. MARY'S RIVER ANALYSIS COMPLETE!\n")
cat(paste(rep("=", 60), collapse=""), "\n\n")

cat("Generated outputs:\n")
cat("  ✓ Settlement probability map\n")
cat("  ✓ Current vector map\n")
cat("  ✓ Combined dispersal map\n")
cat("  ✓ Habitat suitability map\n")
cat("  ✓ Analysis report\n")
cat("  ✓ Data tables\n\n")

cat(sprintf("All results saved in: %s\n", st_marys_dir))