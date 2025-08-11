################################################################################
# Environmental Analysis Functions
# Processing and analyzing environmental variables
################################################################################

#' @title Calculate derived environmental variables
#' @param nc_data NetCDF data object
#' @return Updated nc_data object with derived variables
calculate_derived_variables <- function(nc_data) {
  
  vars <- nc_data$variables
  
  # Calculate current speed from u and v components
  if ("u_surface" %in% names(vars) && "v_surface" %in% names(vars)) {
    vars$current_speed_surface <- sqrt(vars$u_surface^2 + vars$v_surface^2)
    attr(vars$current_speed_surface, "units") <- "m/s"
    attr(vars$current_speed_surface, "long_name") <- "Surface current speed"
  }
  
  if ("u_bottom" %in% names(vars) && "v_bottom" %in% names(vars)) {
    vars$current_speed_bottom <- sqrt(vars$u_bottom^2 + vars$v_bottom^2)
    attr(vars$current_speed_bottom, "units") <- "m/s"
    attr(vars$current_speed_bottom, "long_name") <- "Bottom current speed"
  }
  
  # Calculate current direction
  if ("u_surface" %in% names(vars) && "v_surface" %in% names(vars)) {
    vars$current_direction_surface <- atan2(vars$v_surface, vars$u_surface) * 180/pi
    attr(vars$current_direction_surface, "units") <- "degrees"
    attr(vars$current_direction_surface, "long_name") <- "Surface current direction"
  }
  
  nc_data$variables <- vars
  return(nc_data)
}

#' @title Calculate temporal statistics
#' @param nc_data NetCDF data object
#' @param statistic Character: "mean", "min", "max", "sd"
#' @return List of 2D arrays with temporal statistic for each variable
calculate_temporal_stats <- function(nc_data, statistic = "mean") {
  
  stat_fun <- switch(statistic,
                    "mean" = function(x) mean(x, na.rm = TRUE),
                    "min" = function(x) min(x, na.rm = TRUE),
                    "max" = function(x) max(x, na.rm = TRUE),
                    "sd" = function(x) sd(x, na.rm = TRUE),
                    stop("Invalid statistic"))
  
  results <- list()
  
  for (var_name in names(nc_data$variables)) {
    var_data <- nc_data$variables[[var_name]]
    
    if (length(dim(var_data)) == 3) {
      # Apply statistic across time dimension
      results[[paste0(statistic, "_", var_name)]] <- apply(var_data, c(1, 2), stat_fun)
    } else {
      # Keep 2D variables as is
      results[[var_name]] <- var_data
    }
  }
  
  return(results)
}

#' @title Calculate seasonal patterns
#' @param nc_data NetCDF data object
#' @param spawning_months Integer vector of spawning months
#' @return List with spawning and non-spawning season statistics
analyze_seasonal_patterns <- function(nc_data, spawning_months = 5:9) {
  
  n_months <- length(nc_data$dimensions$time)
  non_spawning_months <- setdiff(1:n_months, spawning_months)
  
  results <- list(
    spawning = list(),
    non_spawning = list(),
    monthly = list()
  )
  
  # Process each variable
  for (var_name in names(nc_data$variables)) {
    var_data <- nc_data$variables[[var_name]]
    
    if (length(dim(var_data)) == 3) {
      # Spawning season mean
      results$spawning[[var_name]] <- apply(var_data[, , spawning_months], 
                                           c(1, 2), mean, na.rm = TRUE)
      
      # Non-spawning season mean
      results$non_spawning[[var_name]] <- apply(var_data[, , non_spawning_months], 
                                               c(1, 2), mean, na.rm = TRUE)
      
      # Monthly means (spatial average)
      mask <- nc_data$variables$mask_land_sea
      monthly_means <- numeric(n_months)
      for (t in 1:n_months) {
        monthly_means[t] <- mean(var_data[, , t][mask == 1], na.rm = TRUE)
      }
      results$monthly[[var_name]] <- monthly_means
    }
  }
  
  return(results)
}

#' @title Calculate Habitat Suitability Index (HSI)
#' @param env_data List with environmental variables (2D arrays)
#' @param optimal_ranges List with optimal ranges for each variable
#' @param mask_land_sea 2D array with land/sea mask
#' @return 2D array with HSI values
calculate_hsi <- function(env_data, optimal_ranges, mask_land_sea) {
  
  # Suitability function for single variable
  calc_suitability <- function(value, opt_range) {
    if (is.na(value)) return(0)
    
    if (value >= opt_range[1] && value <= opt_range[2]) {
      return(1.0)
    } else if (value < opt_range[1]) {
      return(max(0, 1 - (opt_range[1] - value) / diff(opt_range)))
    } else {
      return(max(0, 1 - (value - opt_range[2]) / diff(opt_range)))
    }
  }
  
  # Get dimensions
  dims <- dim(mask_land_sea)
  HSI <- matrix(0, dims[1], dims[2])
  
  # Variables to include in HSI
  hsi_vars <- c("temperature", "salinity", "pH", "O2")
  
  # Calculate HSI for each grid cell
  for (i in 1:dims[1]) {
    for (j in 1:dims[2]) {
      if (mask_land_sea[i, j] == 1) {  # Only water cells
        
        suitabilities <- numeric()
        
        # Check each variable
        for (var in hsi_vars) {
          # Find matching variable in env_data
          var_names <- grep(paste0("^.*", var), names(env_data), 
                           value = TRUE, ignore.case = TRUE)
          
          if (length(var_names) > 0 && var %in% names(optimal_ranges)) {
            value <- env_data[[var_names[1]]][i, j]
            suit <- calc_suitability(value, optimal_ranges[[var]])
            suitabilities <- c(suitabilities, suit)
          }
        }
        
        # Add depth suitability if available
        if ("topography" %in% names(env_data) && "depth" %in% names(optimal_ranges)) {
          depth_suit <- calc_suitability(env_data$topography[i, j], 
                                        optimal_ranges$depth)
          suitabilities <- c(suitabilities, depth_suit)
        }
        
        # Calculate geometric mean
        if (length(suitabilities) > 0) {
          HSI[i, j] <- prod(suitabilities)^(1/length(suitabilities))
        }
      }
    }
  }
  
  return(HSI)
}

#' @title Extract environmental time series at specific locations
#' @param nc_data NetCDF data object
#' @param locations Data frame with lon/lat columns
#' @param variables Character vector of variables to extract
#' @return Data frame with time series for each location
extract_time_series <- function(nc_data, locations, variables = NULL) {
  
  if (is.null(variables)) {
    # Get all 3D variables
    variables <- names(nc_data$variables)[
      sapply(nc_data$variables, function(x) length(dim(x)) == 3)
    ]
  }
  
  # Find grid indices for locations
  indices <- find_reef_grid_indices(locations,
                                   nc_data$dimensions$lon,
                                   nc_data$dimensions$lat)
  
  # Initialize results list
  results <- list()
  
  for (i in 1:nrow(locations)) {
    loc_name <- if ("SourceReef" %in% names(locations)) {
      locations$SourceReef[i]
    } else {
      paste0("Location_", i)
    }
    
    # Extract time series for each variable
    loc_data <- data.frame(
      time = 1:length(nc_data$dimensions$time),
      location = loc_name
    )
    
    for (var in variables) {
      if (var %in% names(nc_data$variables)) {
        var_data <- nc_data$variables[[var]]
        if (length(dim(var_data)) == 3) {
          loc_data[[var]] <- var_data[indices[i, "lon_idx"], 
                                     indices[i, "lat_idx"], ]
        }
      }
    }
    
    results[[loc_name]] <- loc_data
  }
  
  # Combine all locations
  combined <- do.call(rbind, results)
  return(combined)
}

#' @title Calculate environmental gradients
#' @param env_field 2D environmental field
#' @param resolution Grid resolution in km
#' @return List with gradient magnitude and direction
calculate_gradient <- function(env_field, resolution = 1) {
  
  dims <- dim(env_field)
  
  # Calculate gradients using central differences
  dx <- matrix(0, dims[1], dims[2])
  dy <- matrix(0, dims[1], dims[2])
  
  # X-gradient (longitude)
  for (i in 2:(dims[1]-1)) {
    for (j in 1:dims[2]) {
      if (!is.na(env_field[i-1, j]) && !is.na(env_field[i+1, j])) {
        dx[i, j] <- (env_field[i+1, j] - env_field[i-1, j]) / (2 * resolution)
      }
    }
  }
  
  # Y-gradient (latitude)
  for (i in 1:dims[1]) {
    for (j in 2:(dims[2]-1)) {
      if (!is.na(env_field[i, j-1]) && !is.na(env_field[i, j+1])) {
        dy[i, j] <- (env_field[i, j+1] - env_field[i, j-1]) / (2 * resolution)
      }
    }
  }
  
  # Calculate magnitude and direction
  magnitude <- sqrt(dx^2 + dy^2)
  direction <- atan2(dy, dx) * 180/pi
  
  return(list(
    magnitude = magnitude,
    direction = direction,
    dx = dx,
    dy = dy
  ))
}

#' @title Summarize environmental conditions
#' @param env_stats List of environmental statistics
#' @param mask_land_sea Land/sea mask
#' @return Data frame with summary statistics
summarize_environment <- function(env_stats, mask_land_sea) {
  
  summary_df <- data.frame(
    Variable = character(),
    Mean = numeric(),
    SD = numeric(),
    Min = numeric(),
    Max = numeric(),
    stringsAsFactors = FALSE
  )
  
  for (var_name in names(env_stats)) {
    if (is.matrix(env_stats[[var_name]]) || is.array(env_stats[[var_name]])) {
      values <- env_stats[[var_name]][mask_land_sea == 1]
      values <- values[!is.na(values)]
      
      if (length(values) > 0) {
        summary_df <- rbind(summary_df, data.frame(
          Variable = var_name,
          Mean = mean(values),
          SD = sd(values),
          Min = min(values),
          Max = max(values),
          stringsAsFactors = FALSE
        ))
      }
    }
  }
  
  return(summary_df)
}