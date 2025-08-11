################################################################################
# Visualization Utilities
# Functions for creating maps, plots, and visualizations
################################################################################

# Load visualization packages
load_viz_packages <- function() {
  required <- c("ggplot2", "viridis", "RColorBrewer", "gridExtra")
  for (pkg in required) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
      stop(paste("Required package not installed:", pkg))
    }
  }
}

#' @title Create custom theme for plots
#' @param base_size Base font size
#' @return ggplot2 theme object
theme_dispersal <- function(base_size = 11) {
  theme_minimal(base_size = base_size) +
    theme(
      plot.title = element_text(face = "bold", size = base_size * 1.2),
      plot.subtitle = element_text(size = base_size * 0.9),
      axis.title = element_text(size = base_size),
      legend.title = element_text(face = "bold", size = base_size),
      legend.position = "right",
      panel.grid.minor = element_blank(),
      panel.border = element_rect(fill = NA, color = "gray80")
    )
}

#' @title Create environmental variable map
#' @param env_data 2D array of environmental data
#' @param lon Longitude vector
#' @param lat Latitude vector
#' @param reef_data Optional reef data to overlay
#' @param title Plot title
#' @param var_name Variable name for legend
#' @param color_scale Color scale name
#' @param limits Optional limits for color scale
#' @return ggplot object
plot_environmental_map <- function(env_data, lon, lat, 
                                  reef_data = NULL,
                                  title = "Environmental Variable",
                                  var_name = "Value",
                                  color_scale = "viridis",
                                  limits = NULL) {
  
  # Prepare data frame
  df <- expand.grid(lon = lon, lat = lat)
  df$value <- as.vector(env_data)
  df <- df[!is.na(df$value), ]
  
  # Create base plot
  p <- ggplot(df, aes(x = lon, y = lat, fill = value)) +
    geom_tile() +
    coord_fixed(ratio = 1) +
    labs(title = title, x = "Longitude", y = "Latitude") +
    theme_dispersal()
  
  # Apply color scale
  if (color_scale %in% c("viridis", "plasma", "inferno", "magma", "cividis", "mako")) {
    p <- p + scale_fill_viridis(option = color_scale, name = var_name, limits = limits)
  } else if (color_scale == "temperature") {
    p <- p + scale_fill_gradient2(low = "blue", mid = "yellow", high = "red",
                                  midpoint = mean(df$value), name = var_name, 
                                  limits = limits)
  } else if (color_scale == "salinity") {
    p <- p + scale_fill_gradient(low = "lightblue", high = "darkblue",
                                 name = var_name, limits = limits)
  } else {
    p <- p + scale_fill_viridis(name = var_name, limits = limits)
  }
  
  # Add reef points if provided
  if (!is.null(reef_data)) {
    p <- p + geom_point(data = reef_data,
                       aes(x = Longitude, y = Latitude, size = AvgDensity),
                       color = "red", alpha = 0.7, inherit.aes = FALSE) +
      scale_size_continuous(name = "Oyster\nDensity", range = c(2, 8))
  }
  
  return(p)
}

#' @title Create current vector map
#' @param u_field U-component of velocity (2D array)
#' @param v_field V-component of velocity (2D array)
#' @param lon Longitude vector
#' @param lat Latitude vector
#' @param mask Optional land/sea mask
#' @param reef_data Optional reef data to overlay
#' @param skip Subsampling rate for vectors
#' @param scale_factor Scale factor for vector arrows
#' @param title Plot title
#' @return ggplot object
plot_current_vectors <- function(u_field, v_field, lon, lat,
                                mask = NULL, reef_data = NULL,
                                skip = 3, scale_factor = 0.3,
                                title = "Current Vectors") {
  
  # Calculate speed
  speed <- sqrt(u_field^2 + v_field^2)
  
  # Prepare background data
  bg_df <- expand.grid(lon = lon, lat = lat)
  bg_df$speed <- as.vector(speed)
  
  # Apply mask if provided
  if (!is.null(mask)) {
    bg_df$speed[as.vector(mask) == 0] <- NA
  }
  bg_df <- bg_df[!is.na(bg_df$speed), ]
  
  # Prepare vector data (subsampled)
  lon_vec <- lon[seq(1, length(lon), skip)]
  lat_vec <- lat[seq(1, length(lat), skip)]
  u_vec <- u_field[seq(1, nrow(u_field), skip), seq(1, ncol(u_field), skip)]
  v_vec <- v_field[seq(1, nrow(v_field), skip), seq(1, ncol(v_field), skip)]
  
  vec_df <- expand.grid(lon = lon_vec, lat = lat_vec)
  vec_df$u <- as.vector(u_vec)
  vec_df$v <- as.vector(v_vec)
  vec_df$speed <- sqrt(vec_df$u^2 + vec_df$v^2)
  
  # Remove NAs
  vec_df <- vec_df[!is.na(vec_df$u) & !is.na(vec_df$v), ]
  
  # Scale vectors
  vec_df$u_scaled <- vec_df$u * scale_factor
  vec_df$v_scaled <- vec_df$v * scale_factor
  
  # Create plot
  p <- ggplot() +
    geom_tile(data = bg_df, aes(x = lon, y = lat, fill = speed)) +
    scale_fill_viridis(option = "viridis", name = "Speed\n(m/s)") +
    geom_segment(data = vec_df,
                aes(x = lon, y = lat,
                    xend = lon + u_scaled, yend = lat + v_scaled),
                arrow = arrow(length = unit(0.15, "cm"), type = "closed"),
                color = "black", alpha = 0.8, size = 0.5) +
    coord_fixed(ratio = 1) +
    labs(title = title, x = "Longitude", y = "Latitude") +
    theme_dispersal() +
    theme(panel.background = element_rect(fill = "lightgray"),
          panel.grid.major = element_line(color = "white", size = 0.3))
  
  # Add reef points if provided
  if (!is.null(reef_data)) {
    p <- p + geom_point(data = reef_data,
                       aes(x = Longitude, y = Latitude, size = AvgDensity),
                       color = "darkred", fill = "orange", shape = 21, 
                       stroke = 2, alpha = 0.95, inherit.aes = FALSE) +
      scale_size_continuous(name = "Oyster\nDensity", range = c(4, 12))
  }
  
  return(p)
}

#' @title Create connectivity matrix heatmap
#' @param conn_matrix Connectivity matrix
#' @param cluster Logical, whether to cluster rows/columns
#' @param title Plot title
#' @return ggplot object
plot_connectivity_matrix <- function(conn_matrix, cluster = TRUE,
                                    title = "Connectivity Matrix") {
  
  # Convert to long format
  df <- as.data.frame(as.table(conn_matrix))
  names(df) <- c("Source", "Sink", "Connectivity")
  
  # Optionally reorder by clustering
  if (cluster && nrow(conn_matrix) > 2) {
    hc <- hclust(dist(conn_matrix))
    order <- hc$order
    df$Source <- factor(df$Source, levels = rownames(conn_matrix)[order])
    df$Sink <- factor(df$Sink, levels = colnames(conn_matrix)[order])
  }
  
  # Create heatmap
  p <- ggplot(df, aes(x = Sink, y = Source, fill = Connectivity)) +
    geom_tile() +
    scale_fill_gradient2(low = "white", mid = "yellow", high = "darkred",
                        midpoint = 0.5, limits = c(0, 1),
                        name = "Connectivity") +
    labs(title = title, x = "Sink Reef", y = "Source Reef") +
    theme_dispersal() +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5, size = 7),
          axis.text.y = element_text(size = 7))
  
  return(p)
}

#' @title Create network classification plot
#' @param metrics Network metrics data frame
#' @param title Plot title
#' @return ggplot object
plot_network_classification <- function(metrics, 
                                       title = "Reef Network Classification") {
  
  p <- ggplot(metrics, aes(x = OutStrength, y = InStrength)) +
    geom_point(aes(size = Density, color = Type), alpha = 0.7) +
    geom_text(aes(label = SourceReef), size = 2.5, vjust = -0.5) +
    scale_color_manual(values = c("Source" = "blue", "Sink" = "red",
                                 "Hub" = "green", "Isolated" = "gray")) +
    scale_size_continuous(name = "Oyster\nDensity", range = c(3, 10)) +
    labs(title = title,
         x = "Export Strength (Source)",
         y = "Import Strength (Sink)") +
    theme_dispersal()
  
  # Add quadrant lines
  p <- p + 
    geom_hline(yintercept = median(metrics$InStrength), 
              linetype = "dashed", alpha = 0.3) +
    geom_vline(xintercept = median(metrics$OutStrength), 
              linetype = "dashed", alpha = 0.3)
  
  return(p)
}

#' @title Create temporal variation plot
#' @param monthly_data Data frame with monthly statistics
#' @param variables Character vector of variables to plot
#' @param title Plot title
#' @return ggplot object
plot_temporal_variation <- function(monthly_data, 
                                   variables = c("Temperature", "Salinity"),
                                   title = "Monthly Environmental Variation") {
  
  # Reshape data to long format
  df_long <- monthly_data %>%
    select(Month, all_of(variables)) %>%
    pivot_longer(cols = -Month, names_to = "Variable", values_to = "Value")
  
  # Create plot
  p <- ggplot(df_long, aes(x = Month, y = Value, color = Variable)) +
    geom_line(size = 1.2) +
    geom_point(size = 3) +
    facet_wrap(~Variable, scales = "free_y", ncol = 2) +
    scale_x_continuous(breaks = 1:12, labels = month.abb) +
    labs(title = title, x = "Month", y = "Value") +
    theme_dispersal() +
    theme(legend.position = "none")
  
  return(p)
}

#' @title Create settlement probability map
#' @param settlement_field 2D array of settlement probabilities
#' @param lon Longitude vector
#' @param lat Latitude vector
#' @param reef_data Source reef data
#' @param title Plot title
#' @return ggplot object
plot_settlement_probability <- function(settlement_field, lon, lat, reef_data,
                                       title = "Larval Settlement Probability") {
  
  # Prepare data
  df <- expand.grid(lon = lon, lat = lat)
  df$probability <- as.vector(settlement_field)
  df <- df[!is.na(df$probability) & df$probability > 0, ]
  
  # Create plot
  p <- ggplot() +
    geom_tile(data = df, aes(x = lon, y = lat, fill = probability)) +
    scale_fill_gradientn(colors = c("darkblue", "blue", "lightblue", "yellow", "orange", "darkred"),
                        values = c(0, 0.1, 0.3, 0.5, 0.7, 1),
                        name = "Settlement\nProbability",
                        limits = c(0, 1)) +
    geom_point(data = reef_data,
              aes(x = Longitude, y = Latitude, size = AvgDensity),
              color = "black", fill = "yellow", shape = 21, 
              stroke = 2.5, alpha = 1) +
    scale_size_continuous(name = "Oyster\nDensity", range = c(4, 12)) +
    coord_fixed(ratio = 1) +
    labs(title = title, x = "Longitude", y = "Latitude") +
    theme_dispersal() +
    theme(panel.background = element_rect(fill = "white"),
          panel.grid.major = element_line(color = "gray90", size = 0.3))
  
  return(p)
}

#' @title Create multi-panel figure
#' @param plots List of ggplot objects
#' @param ncol Number of columns
#' @param title Overall title
#' @return Combined plot
create_multi_panel <- function(plots, ncol = 2, title = NULL) {
  
  combined <- gridExtra::grid.arrange(
    grobs = plots,
    ncol = ncol,
    top = title
  )
  
  return(combined)
}

#' @title Save plot with consistent settings
#' @param plot ggplot object or grob
#' @param filename Output filename
#' @param width Width in inches
#' @param height Height in inches
#' @param dpi Resolution
save_plot <- function(plot, filename, width = 10, height = 8, dpi = 300) {
  
  # Create directory if needed
  dir <- dirname(filename)
  if (!dir.exists(dir)) {
    dir.create(dir, recursive = TRUE)
  }
  
  # Save plot
  ggsave(filename, plot, width = width, height = height, dpi = dpi)
  
  message(paste("Plot saved:", filename))
}