################################################################################
# Dispersal Modeling Module - Fixed Version
# Biophysical larval dispersal with advection and diffusion
################################################################################

#' @title Calculate Haversine distance between two points
#' @param lon1 Longitude of point 1 (degrees)
#' @param lat1 Latitude of point 1 (degrees)
#' @param lon2 Longitude of point 2 (degrees)
#' @param lat2 Latitude of point 2 (degrees)
#' @return Distance in kilometers
haversine_distance <- function(lon1, lat1, lon2, lat2) {
  R <- 6371  # Earth radius in km
  
  # Convert to radians
  lon1_rad <- lon1 * pi / 180
  lat1_rad <- lat1 * pi / 180
  lon2_rad <- lon2 * pi / 180
  lat2_rad <- lat2 * pi / 180
  
  # Haversine formula
  dlon <- lon2_rad - lon1_rad
  dlat <- lat2_rad - lat1_rad
  a <- sin(dlat/2)^2 + cos(lat1_rad) * cos(lat2_rad) * sin(dlon/2)^2
  c <- 2 * asin(sqrt(a))
  
  return(R * c)
}

#' @title Calculate distance matrix between all reef pairs
#' @param reef_data Data frame with Longitude and Latitude columns
#' @return Distance matrix in kilometers
calculate_distance_matrix <- function(reef_data) {
  n_reefs <- nrow(reef_data)
  dist_matrix <- matrix(0, n_reefs, n_reefs)
  
  for (i in 1:n_reefs) {
    for (j in 1:n_reefs) {
      dist_matrix[i, j] <- haversine_distance(
        reef_data$Longitude[i], reef_data$Latitude[i],
        reef_data$Longitude[j], reef_data$Latitude[j]
      )
    }
  }
  
  rownames(dist_matrix) <- reef_data$SourceReef
  colnames(dist_matrix) <- reef_data$SourceReef
  
  return(dist_matrix)
}

#' @title Calculate single source-sink connectivity
#' @description Models larval transport from source to sink considering:
#'   1. Advection by currents (directional transport)
#'   2. Diffusion (random spread)
#'   3. Mortality (exponential decay)
#'   4. Distance limitations
#' @param source_idx Source reef grid indices [lat_idx, lon_idx]
#' @param sink_idx Sink reef grid indices [lat_idx, lon_idx]  
#' @param distance Direct distance between reefs (km)
#' @param u_field U-velocity field [lon, lat, time] in m/s
#' @param v_field V-velocity field [lon, lat, time] in m/s
#' @param params List with dispersal parameters
#' @param source_lon Source longitude for drift calculation
#' @param source_lat Source latitude for drift calculation
#' @param sink_lon Sink longitude for drift calculation
#' @param sink_lat Sink latitude for drift calculation
#' @return Connectivity probability (0-1)
calculate_single_connectivity <- function(source_idx, sink_idx, distance,
                                         u_field, v_field, params,
                                         source_lon, source_lat, 
                                         sink_lon, sink_lat) {
  
  # No connectivity beyond maximum distance
  if (distance > params$max_dispersal_distance) {
    return(0)
  }
  
  # Extract currents at source location
  # Arrays are [lon, lat, time], indices are [lat_idx, lon_idx]
  # So access as: array[lon_idx, lat_idx, time]
  u_at_source <- u_field[source_idx[2], source_idx[1], ]
  v_at_source <- v_field[source_idx[2], source_idx[1], ]
  
  # Calculate mean current velocity at source
  u_mean <- mean(u_at_source, na.rm = TRUE)
  v_mean <- mean(v_at_source, na.rm = TRUE)
  
  # Handle missing current data
  if (is.na(u_mean)) u_mean <- 0
  if (is.na(v_mean)) v_mean <- 0
  
  # Calculate larval drift due to currents
  # Convert m/s to degrees per day
  meters_per_degree_lon <- 111000 * cos(source_lat * pi/180)
  meters_per_degree_lat <- 111000
  
  u_deg_per_day <- (u_mean * 86400) / meters_per_degree_lon
  v_deg_per_day <- (v_mean * 86400) / meters_per_degree_lat
  
  # Where larvae end up after drifting with current
  drift_lon <- source_lon + (u_deg_per_day * params$pelagic_larval_duration)
  drift_lat <- source_lat + (v_deg_per_day * params$pelagic_larval_duration)
  
  # Distance from drift endpoint to sink
  drift_to_sink_distance <- haversine_distance(drift_lon, drift_lat, sink_lon, sink_lat)
  
  # Diffusive spread (2D random walk)
  # Standard deviation of spread after PLD
  diffusion_km <- sqrt(2 * params$diffusion_coefficient * 
                      params$pelagic_larval_duration * 86400) / 1000
  
  # Probability of reaching sink from drift endpoint (2D Gaussian)
  # Account for settlement competency window
  if (params$settlement_competency > 0) {
    settlement_window <- params$pelagic_larval_duration - params$settlement_competency
    if (settlement_window <= 0) settlement_window <- 1
  } else {
    settlement_window <- params$pelagic_larval_duration
  }
  
  # Gaussian probability with appropriate variance
  dispersal_prob <- exp(-drift_to_sink_distance^2 / (2 * diffusion_km^2))
  
  # Survival probability after PLD with daily mortality
  survival_prob <- (1 - params$mortality_rate)^params$pelagic_larval_duration
  
  # Combined probability
  connectivity <- dispersal_prob * survival_prob
  
  # Apply small boost for self-recruitment (eddies, retention zones)
  if (distance < 1) {  # Same reef or very close
    connectivity <- connectivity * 1.5
  }
  
  # Ensure probability stays in [0,1]
  return(min(1, connectivity))
}

#' @title Build connectivity matrix
#' @param reef_data Data frame with reef information
#' @param nc_data NetCDF data object with currents
#' @param params Dispersal parameters
#' @param verbose Show progress bar
#' @return Connectivity matrix (rows=sources, cols=sinks)
build_connectivity_matrix <- function(reef_data, nc_data, params, verbose = TRUE) {
  
  n_reefs <- nrow(reef_data)
  
  # Get reef grid indices
  reef_indices <- find_reef_grid_indices(
    reef_data,
    nc_data$dimensions$lon,
    nc_data$dimensions$lat
  )
  
  # Calculate distances
  dist_matrix <- calculate_distance_matrix(reef_data)
  
  # Get current fields
  u_field <- nc_data$variables$u_surface
  v_field <- nc_data$variables$v_surface
  
  # Initialize connectivity matrix
  conn_matrix <- matrix(0, n_reefs, n_reefs)
  
  # Progress bar
  if (verbose) {
    pb <- txtProgressBar(min = 0, max = n_reefs, style = 3)
  }
  
  # Calculate connectivity for each source-sink pair
  for (i in 1:n_reefs) {
    for (j in 1:n_reefs) {
      
      # Base connectivity from dispersal model
      base_conn <- calculate_single_connectivity(
        source_idx = reef_indices[i, ],
        sink_idx = reef_indices[j, ],
        distance = dist_matrix[i, j],
        u_field = u_field,
        v_field = v_field,
        params = params,
        source_lon = reef_data$Longitude[i],
        source_lat = reef_data$Latitude[i],
        sink_lon = reef_data$Longitude[j],
        sink_lat = reef_data$Latitude[j]
      )
      
      # Scale by source fecundity (larvae produced)
      # Use sqrt to reduce dominance of high-density reefs
      larvae_production <- sqrt(reef_data$AvgDensity[i])
      
      conn_matrix[i, j] <- base_conn * larvae_production
    }
    
    if (verbose) {
      setTxtProgressBar(pb, i)
    }
  }
  
  if (verbose) {
    close(pb)
    cat("\n")
  }
  
  # Normalize each row to sum to 1 (proportion of larvae from each source)
  # This maintains relative patterns while making interpretation easier
  for (i in 1:n_reefs) {
    row_sum <- sum(conn_matrix[i, ])
    if (row_sum > 0) {
      conn_matrix[i, ] <- conn_matrix[i, ] / row_sum
    }
  }
  
  # Add reef names
  rownames(conn_matrix) <- reef_data$SourceReef
  colnames(conn_matrix) <- reef_data$SourceReef
  
  return(conn_matrix)
}

#' @title Calculate settlement probability field
#' @param source_reefs Data frame with source reef information
#' @param nc_data NetCDF data object
#' @param params Dispersal parameters
#' @return 2D settlement probability field
calculate_settlement_field <- function(source_reefs, nc_data, params) {
  
  lon <- nc_data$dimensions$lon
  lat <- nc_data$dimensions$lat
  n_lon <- length(lon)
  n_lat <- length(lat)
  
  # Initialize field
  settlement_field <- matrix(0, n_lon, n_lat)
  
  # Get mean currents
  u_mean <- apply(nc_data$variables$u_surface, c(1, 2), mean, na.rm = TRUE)
  v_mean <- apply(nc_data$variables$v_surface, c(1, 2), mean, na.rm = TRUE)
  
  # Get reef grid indices
  reef_indices <- find_reef_grid_indices(
    source_reefs,
    nc_data$dimensions$lon,
    nc_data$dimensions$lat
  )
  
  # Add contribution from each source
  for (i in 1:nrow(source_reefs)) {
    
    # Get source location and currents
    source_lon <- source_reefs$Longitude[i]
    source_lat <- source_reefs$Latitude[i]
    source_idx <- reef_indices[i, ]
    
    # Current at source (correct indexing)
    u_source <- u_mean[source_idx[2], source_idx[1]]
    v_source <- v_mean[source_idx[2], source_idx[1]]
    
    if (is.na(u_source)) u_source <- 0
    if (is.na(v_source)) v_source <- 0
    
    # Calculate drift center
    meters_per_degree_lon <- 111000 * cos(source_lat * pi/180)
    meters_per_degree_lat <- 111000
    
    drift_lon <- source_lon + (u_source * params$pelagic_larval_duration * 86400) / meters_per_degree_lon
    drift_lat <- source_lat + (v_source * params$pelagic_larval_duration * 86400) / meters_per_degree_lat
    
    # Diffusion spread
    spread_km <- sqrt(2 * params$diffusion_coefficient * 
                     params$pelagic_larval_duration * 86400) / 1000
    
    # Add Gaussian plume
    for (j in 1:n_lon) {
      for (k in 1:n_lat) {
        dist <- haversine_distance(lon[j], lat[k], drift_lon, drift_lat)
        
        # Gaussian with mortality
        contribution <- exp(-dist^2 / (2 * spread_km^2)) * 
                       (1 - params$mortality_rate)^params$pelagic_larval_duration
        
        # Scale by source strength
        settlement_field[j, k] <- settlement_field[j, k] + 
                                 contribution * sqrt(source_reefs$AvgDensity[i])
      }
    }
  }
  
  # Apply land mask
  if ("mask_land_sea" %in% names(nc_data$variables)) {
    settlement_field <- settlement_field * nc_data$variables$mask_land_sea
  }
  
  # Normalize for visualization
  max_val <- max(settlement_field, na.rm = TRUE)
  if (max_val > 0) {
    settlement_field <- settlement_field / max_val
  }
  
  return(settlement_field)
}

#' @title Calculate network metrics for each reef
#' @param conn_matrix Connectivity matrix
#' @param reef_data Data frame with reef information
#' @return Data frame with network metrics
calculate_network_metrics <- function(conn_matrix, reef_data) {
  
  n_reefs <- nrow(conn_matrix)
  
  # Initialize metrics data frame
  metrics <- data.frame(
    SourceReef = reef_data$SourceReef,
    Longitude = reef_data$Longitude,
    Latitude = reef_data$Latitude,
    Density = reef_data$AvgDensity,
    stringsAsFactors = FALSE
  )
  
  # Connectivity threshold
  threshold <- 0.01
  
  # Degree (number of connections)
  metrics$OutDegree <- rowSums(conn_matrix > threshold)
  metrics$InDegree <- colSums(conn_matrix > threshold)
  
  # Strength (sum of connections)
  metrics$OutStrength <- rowSums(conn_matrix)
  metrics$InStrength <- colSums(conn_matrix)
  
  # Self-recruitment
  metrics$SelfRecruitment <- diag(conn_matrix)
  
  # Local retention ratio
  metrics$LocalRetention <- diag(conn_matrix) / rowSums(conn_matrix)
  metrics$LocalRetention[is.na(metrics$LocalRetention)] <- 0
  
  # Net flux
  metrics$NetExport <- metrics$OutStrength - metrics$InStrength
  
  # Betweenness centrality (simplified)
  betweenness <- numeric(n_reefs)
  
  # Convert connectivity to distance for shortest path
  dist_for_path <- -log(conn_matrix + 1e-10)
  dist_for_path[!is.finite(dist_for_path)] <- 1000
  diag(dist_for_path) <- 0
  
  # Count shortest paths through each node
  for (s in 1:n_reefs) {
    for (t in 1:n_reefs) {
      if (s != t) {
        for (v in 1:n_reefs) {
          if (v != s && v != t) {
            # Check if path through v is shorter
            if (dist_for_path[s,v] + dist_for_path[v,t] < dist_for_path[s,t] * 0.9) {
              betweenness[v] <- betweenness[v] + 1
            }
          }
        }
      }
    }
  }
  
  metrics$Betweenness <- betweenness
  
  # Classify reefs based on metrics
  metrics$Type <- "Balanced"
  
  # Sources: high net export
  source_threshold <- quantile(metrics$NetExport, 0.75)
  metrics$Type[metrics$NetExport > source_threshold] <- "Source"
  
  # Sinks: high net import
  sink_threshold <- quantile(metrics$NetExport, 0.25)
  metrics$Type[metrics$NetExport < sink_threshold] <- "Sink"
  
  # Hubs: high betweenness
  hub_threshold <- quantile(metrics$Betweenness, 0.75)
  metrics$Type[metrics$Betweenness > hub_threshold & 
               metrics$Type == "Balanced"] <- "Hub"
  
  # Isolated: low connectivity
  isolated_threshold <- quantile(metrics$OutDegree + metrics$InDegree, 0.25)
  metrics$Type[(metrics$OutDegree + metrics$InDegree) < isolated_threshold &
               metrics$Type == "Balanced"] <- "Isolated"
  
  return(metrics)
}

#' @title Assess connectivity patterns
#' @param conn_matrix Connectivity matrix
#' @return List of connectivity statistics
assess_connectivity <- function(conn_matrix) {
  
  # Off-diagonal elements
  conn_no_diag <- conn_matrix
  diag(conn_no_diag) <- NA
  
  # Calculate statistics
  stats <- list(
    mean_connectivity = mean(conn_no_diag, na.rm = TRUE),
    median_connectivity = median(conn_no_diag, na.rm = TRUE),
    sd_connectivity = sd(conn_no_diag, na.rm = TRUE),
    max_connectivity = max(conn_no_diag, na.rm = TRUE),
    n_connections = sum(conn_no_diag > 0.01, na.rm = TRUE),
    n_strong = sum(conn_no_diag > 0.1, na.rm = TRUE),
    mean_self_recruitment = mean(diag(conn_matrix)),
    proportion_connected = sum(conn_no_diag > 0.01, na.rm = TRUE) / 
                          (length(conn_no_diag) - nrow(conn_matrix))
  )
  
  return(stats)
}