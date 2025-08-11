################################################################################
# Data Loading Utilities
# Functions for loading and preprocessing NetCDF and Excel data
################################################################################

#' @title Load required packages
load_required_packages <- function() {
  required_packages <- c("ncdf4", "readxl", "dplyr", "tidyr")
  
  for (pkg in required_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      stop(paste("Required package not installed:", pkg))
    }
  }
  invisible(TRUE)
}

#' @title Load NetCDF data
#' @param nc_file Path to NetCDF file
#' @param variables Character vector of variables to load (NULL for all)
#' @param subset_bounds List with lon_min, lon_max, lat_min, lat_max (NULL for full)
#' @return List containing dimensions and variables
load_netcdf_data <- function(nc_file, variables = NULL, subset_bounds = NULL) {
  
  if (!file.exists(nc_file)) {
    stop(paste("NetCDF file not found:", nc_file))
  }
  
  nc <- nc_open(nc_file)
  
  # Get dimensions
  lon <- ncvar_get(nc, "longitude")
  lat <- ncvar_get(nc, "latitude")
  time <- ncvar_get(nc, "time")
  
  # Apply spatial subset if requested
  if (!is.null(subset_bounds)) {
    lon_idx <- which(lon >= subset_bounds$lon_min & lon <= subset_bounds$lon_max)
    lat_idx <- which(lat >= subset_bounds$lat_min & lat <= subset_bounds$lat_max)
  } else {
    lon_idx <- 1:length(lon)
    lat_idx <- 1:length(lat)
  }
  
  # Subset dimensions
  lon_subset <- lon[lon_idx]
  lat_subset <- lat[lat_idx]
  
  # Determine which variables to load
  if (is.null(variables)) {
    var_names <- names(nc$var)
    # Exclude dimension variables
    var_names <- var_names[!var_names %in% c("longitude", "latitude", "time", "crs")]
  } else {
    var_names <- variables
  }
  
  # Load variables
  data_vars <- list()
  
  for (var_name in var_names) {
    if (var_name %in% names(nc$var)) {
      var_data <- ncvar_get(nc, var_name)
      
      # Handle different dimensionalities
      var_dims <- nc$var[[var_name]]$ndims
      
      if (var_dims == 3) {
        # 3D variable [lon, lat, time]
        data_vars[[var_name]] <- var_data[lon_idx, lat_idx, ]
      } else if (var_dims == 2) {
        # 2D variable [lon, lat]
        data_vars[[var_name]] <- var_data[lon_idx, lat_idx]
      } else {
        # 1D or other
        data_vars[[var_name]] <- var_data
      }
      
      # Store attributes
      attr(data_vars[[var_name]], "units") <- ncatt_get(nc, var_name, "units")$value
      attr(data_vars[[var_name]], "long_name") <- ncatt_get(nc, var_name, "long_name")$value
    } else {
      warning(paste("Variable not found in NetCDF:", var_name))
    }
  }
  
  nc_close(nc)
  
  # Return structured data
  result <- list(
    dimensions = list(
      lon = lon_subset,
      lat = lat_subset,
      time = time,
      original_indices = list(lon = lon_idx, lat = lat_idx)
    ),
    variables = data_vars,
    metadata = list(
      file = nc_file,
      n_variables = length(data_vars),
      spatial_extent = c(
        lon_min = min(lon_subset),
        lon_max = max(lon_subset),
        lat_min = min(lat_subset),
        lat_max = max(lat_subset)
      )
    )
  )
  
  class(result) <- c("netcdf_data", "list")
  return(result)
}

#' @title Load oyster reef data
#' @param excel_file Path to Excel file with reef coordinates
#' @param filter_prefix Character prefix to filter reefs (e.g., "STM" for St. Mary's)
#' @return Data frame with reef information
load_reef_data <- function(excel_file, filter_prefix = NULL) {
  
  if (!file.exists(excel_file)) {
    stop(paste("Excel file not found:", excel_file))
  }
  
  # Load data
  reef_data <- read_excel(excel_file)
  
  # Validate required columns
  required_cols <- c("SourceReef", "Longitude", "Latitude", "AvgDensity")
  missing_cols <- setdiff(required_cols, names(reef_data))
  
  if (length(missing_cols) > 0) {
    stop(paste("Missing required columns:", paste(missing_cols, collapse = ", ")))
  }
  
  # Apply filter if requested
  if (!is.null(filter_prefix)) {
    reef_data <- reef_data %>%
      filter(grepl(filter_prefix, SourceReef))
  }
  
  # Add calculated fields
  reef_data <- reef_data %>%
    mutate(
      reef_id = 1:n(),
      density_category = cut(AvgDensity, 
                            breaks = c(0, 50, 100, 200, Inf),
                            labels = c("Low", "Medium", "High", "Very High"))
    )
  
  # Add metadata as attributes
  attr(reef_data, "source_file") <- excel_file
  attr(reef_data, "n_reefs") <- nrow(reef_data)
  attr(reef_data, "total_density") <- sum(reef_data$AvgDensity)
  attr(reef_data, "spatial_extent") <- list(
    lon_min = min(reef_data$Longitude),
    lon_max = max(reef_data$Longitude),
    lat_min = min(reef_data$Latitude),
    lat_max = max(reef_data$Latitude)
  )
  
  class(reef_data) <- c("reef_data", class(reef_data))
  return(reef_data)
}

#' @title Find nearest grid points for reef locations
#' @param reef_data Data frame with reef coordinates
#' @param lon_grid Longitude grid vector
#' @param lat_grid Latitude grid vector
#' @return Matrix with grid indices for each reef
find_reef_grid_indices <- function(reef_data, lon_grid, lat_grid) {
  
  n_reefs <- nrow(reef_data)
  indices <- matrix(NA, nrow = n_reefs, ncol = 2,
                   dimnames = list(reef_data$SourceReef, c("lat_idx", "lon_idx")))
  
  for (i in 1:n_reefs) {
    lon_idx <- which.min(abs(lon_grid - reef_data$Longitude[i]))
    lat_idx <- which.min(abs(lat_grid - reef_data$Latitude[i]))
    indices[i, ] <- c(lat_idx, lon_idx)
  }
  
  return(indices)
}

#' @title Extract environmental conditions at reef locations
#' @param reef_data Data frame with reef information
#' @param nc_data NetCDF data object from load_netcdf_data()
#' @param variables Character vector of variables to extract
#' @return Data frame with environmental conditions at each reef
extract_reef_environment <- function(reef_data, nc_data, 
                                    variables = c("temperature_surface", "salinity_surface",
                                                "u_surface", "v_surface", "pH_surface")) {
  
  # Find grid indices
  indices <- find_reef_grid_indices(reef_data, 
                                   nc_data$dimensions$lon,
                                   nc_data$dimensions$lat)
  
  # Initialize results
  env_data <- reef_data
  
  # Extract each variable
  for (var_name in variables) {
    if (var_name %in% names(nc_data$variables)) {
      var_data <- nc_data$variables[[var_name]]
      
      # Calculate temporal mean if 3D
      if (length(dim(var_data)) == 3) {
        var_mean <- apply(var_data, c(1, 2), mean, na.rm = TRUE)
      } else {
        var_mean <- var_data
      }
      
      # Extract values at reef locations
      values <- numeric(nrow(reef_data))
      for (i in 1:nrow(reef_data)) {
        values[i] <- var_mean[indices[i, "lon_idx"], indices[i, "lat_idx"]]
      }
      
      # Add to data frame with clean name
      clean_name <- gsub("_surface|_bottom", "", var_name)
      env_data[[clean_name]] <- values
    }
  }
  
  return(env_data)
}

#' @title Print summary of loaded data
#' @param x Object to summarize
print.netcdf_data <- function(x, ...) {
  cat("NetCDF Data Object\n")
  cat("==================\n")
  cat(sprintf("Source: %s\n", x$metadata$file))
  cat(sprintf("Grid: %d x %d (lon x lat)\n", 
              length(x$dimensions$lon), length(x$dimensions$lat)))
  cat(sprintf("Time steps: %d\n", length(x$dimensions$time)))
  cat(sprintf("Variables loaded: %d\n", x$metadata$n_variables))
  cat(sprintf("Spatial extent: %.2f°W to %.2f°W, %.2f°N to %.2f°N\n",
              abs(x$metadata$spatial_extent["lon_min"]),
              abs(x$metadata$spatial_extent["lon_max"]),
              x$metadata$spatial_extent["lat_min"],
              x$metadata$spatial_extent["lat_max"]))
  cat("\nVariables:\n")
  for (var in names(x$variables)[1:min(10, length(x$variables))]) {
    cat(sprintf("  - %s (%s)\n", var, paste(dim(x$variables[[var]]), collapse="x")))
  }
  if (length(x$variables) > 10) {
    cat(sprintf("  ... and %d more\n", length(x$variables) - 10))
  }
}

print.reef_data <- function(x, ...) {
  cat("Reef Data Object\n")
  cat("================\n")
  cat(sprintf("Source: %s\n", attr(x, "source_file")))
  cat(sprintf("Number of reefs: %d\n", attr(x, "n_reefs")))
  cat(sprintf("Total density: %.1f\n", attr(x, "total_density")))
  cat(sprintf("Mean density: %.1f ± %.1f\n", 
              mean(x$AvgDensity), sd(x$AvgDensity)))
  extent <- attr(x, "spatial_extent")
  cat(sprintf("Spatial extent: %.3f°W to %.3f°W, %.3f°N to %.3f°N\n",
              abs(extent$lon_min), abs(extent$lon_max),
              extent$lat_min, extent$lat_max))
  cat("\nTop 5 reefs by density:\n")
  top_reefs <- head(x[order(x$AvgDensity, decreasing = TRUE), 
                     c("SourceReef", "AvgDensity")], 5)
  print(as.data.frame(top_reefs), row.names = FALSE)
  invisible(x)
}