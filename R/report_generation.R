################################################################################
# Report Generation Functions
# Creating formatted reports and summaries
################################################################################

#' @title Generate analysis report header
#' @param title Report title
#' @param author Report author
#' @param date Date (default: current date)
#' @return Character string with formatted header
generate_header <- function(title, author = "R Analysis Pipeline", 
                           date = Sys.Date()) {
  
  header <- paste0(
    paste(rep("=", 80), collapse = ""), "\n",
    "    ", title, "\n",
    paste(rep("=", 80), collapse = ""), "\n\n",
    "Date: ", format(date, "%Y-%m-%d"), "\n",
    "Author: ", author, "\n\n"
  )
  
  return(header)
}

#' @title Format section header
#' @param title Section title
#' @param level Header level (1, 2, or 3)
#' @return Formatted section header
format_section <- function(title, level = 1) {
  
  if (level == 1) {
    separator <- paste(rep("=", nchar(title)), collapse = "")
    return(paste0("\n", title, "\n", separator, "\n"))
  } else if (level == 2) {
    separator <- paste(rep("-", nchar(title)), collapse = "")
    return(paste0("\n", title, "\n", separator, "\n"))
  } else {
    return(paste0("\n", title, "\n"))
  }
}

#' @title Generate environmental summary
#' @param env_summary Environmental summary data
#' @param monthly_data Monthly statistics
#' @return Formatted text summary
generate_env_summary <- function(env_summary, monthly_data = NULL) {
  
  text <- format_section("ENVIRONMENTAL CONDITIONS", 1)
  
  # Annual statistics
  text <- paste0(text, "\nAnnual Statistics:\n")
  
  for (i in 1:nrow(env_summary)) {
    var <- env_summary$Variable[i]
    text <- paste0(text, sprintf("  • %s: %.2f ± %.2f (range: %.2f - %.2f)\n",
                                var, env_summary$Mean[i], env_summary$SD[i],
                                env_summary$Min[i], env_summary$Max[i]))
  }
  
  # Seasonal patterns
  if (!is.null(monthly_data)) {
    text <- paste0(text, "\nSeasonal Patterns:\n")
    
    # Find peak months
    temp_col <- grep("temp", names(monthly_data), ignore.case = TRUE)[1]
    if (!is.na(temp_col)) {
      peak_temp <- which.max(monthly_data[[temp_col]])
      text <- paste0(text, sprintf("  • Peak temperature: %s (%.1f°C)\n",
                                  month.abb[peak_temp], 
                                  monthly_data[[temp_col]][peak_temp]))
    }
  }
  
  return(text)
}

#' @title Generate connectivity summary
#' @param conn_assessment Connectivity assessment from assess_connectivity()
#' @param metrics Network metrics data frame
#' @return Formatted text summary
generate_connectivity_summary <- function(conn_assessment, metrics) {
  
  text <- format_section("CONNECTIVITY PATTERNS", 1)
  
  # Overall connectivity
  text <- paste0(text, "\nConnectivity Statistics:\n")
  text <- paste0(text, sprintf("  • Mean connectivity: %.4f\n", 
                              conn_assessment$mean_connectivity))
  text <- paste0(text, sprintf("  • Maximum connectivity: %.4f\n", 
                              conn_assessment$max_connectivity))
  text <- paste0(text, sprintf("  • Proportion connected: %.1f%%\n", 
                              conn_assessment$prop_connected * 100))
  
  # Connection strength distribution
  text <- paste0(text, "\nConnection Strength:\n")
  text <- paste0(text, sprintf("  • Strong (>0.5): %d connections\n", 
                              conn_assessment$n_strong))
  text <- paste0(text, sprintf("  • Moderate (0.1-0.5): %d connections\n", 
                              conn_assessment$n_moderate))
  text <- paste0(text, sprintf("  • Weak (<0.1): %d connections\n", 
                              conn_assessment$n_weak))
  
  # Self-recruitment
  text <- paste0(text, sprintf("\nMean self-recruitment rate: %.3f\n", 
                              conn_assessment$mean_self_recruitment))
  
  # Network classification
  text <- paste0(text, "\nReef Classification:\n")
  type_counts <- table(metrics$Type)
  for (type in names(type_counts)) {
    text <- paste0(text, sprintf("  • %s reefs: %d (%.1f%%)\n",
                                type, type_counts[type],
                                type_counts[type]/nrow(metrics) * 100))
  }
  
  return(text)
}

#' @title Generate key reefs summary
#' @param metrics Network metrics data frame
#' @param n_top Number of top reefs to show
#' @return Formatted text summary
generate_key_reefs_summary <- function(metrics, n_top = 5) {
  
  text <- format_section("KEY REEFS", 1)
  
  # Top sources
  text <- paste0(text, "\nTop Source Reefs (Larval Exporters):\n")
  top_sources <- head(metrics[order(metrics$OutStrength, decreasing = TRUE), ], n_top)
  
  for (i in 1:min(n_top, nrow(top_sources))) {
    text <- paste0(text, sprintf("  %d. %s: Export=%.3f, Density=%.1f\n",
                                i, top_sources$SourceReef[i],
                                top_sources$OutStrength[i],
                                top_sources$Density[i]))
  }
  
  # Top sinks
  text <- paste0(text, "\nTop Sink Reefs (Larval Importers):\n")
  top_sinks <- head(metrics[order(metrics$InStrength, decreasing = TRUE), ], n_top)
  
  for (i in 1:min(n_top, nrow(top_sinks))) {
    text <- paste0(text, sprintf("  %d. %s: Import=%.3f, Density=%.1f\n",
                                i, top_sinks$SourceReef[i],
                                top_sinks$InStrength[i],
                                top_sinks$Density[i]))
  }
  
  # Top hubs
  text <- paste0(text, "\nTop Hub Reefs (High Connectivity):\n")
  top_hubs <- head(metrics[order(metrics$Betweenness, decreasing = TRUE), ], n_top)
  
  for (i in 1:min(n_top, nrow(top_hubs))) {
    text <- paste0(text, sprintf("  %d. %s: Betweenness=%.1f, Type=%s\n",
                                i, top_hubs$SourceReef[i],
                                top_hubs$Betweenness[i],
                                top_hubs$Type[i]))
  }
  
  return(text)
}

#' @title Generate management recommendations
#' @param metrics Network metrics
#' @param hsi_stats HSI statistics
#' @param spawning_months Spawning month indices
#' @return Formatted recommendations text
generate_recommendations <- function(metrics, hsi_stats = NULL, 
                                    spawning_months = 5:9) {
  
  text <- format_section("MANAGEMENT RECOMMENDATIONS", 1)
  
  # Priority 1: Protect source reefs
  text <- paste0(text, "\nPRIORITY 1 - Protect Key Source Reefs:\n")
  top_sources <- head(metrics[order(metrics$OutStrength, decreasing = TRUE), 
                             "SourceReef"], 3)
  text <- paste0(text, sprintf("  • Focus on: %s\n", 
                              paste(top_sources, collapse = ", ")))
  text <- paste0(text, "  • These reefs contribute most to regional larval supply\n")
  
  # Priority 2: Restoration
  text <- paste0(text, "\nPRIORITY 2 - Restore High-Suitability Areas:\n")
  if (!is.null(hsi_stats)) {
    text <- paste0(text, sprintf("  • %.1f%% of area has HSI > 0.8\n", 
                                hsi_stats$high_suitability_percent))
  }
  text <- paste0(text, "  • Target restoration in optimal habitat zones\n")
  
  # Priority 3: Connectivity
  text <- paste0(text, "\nPRIORITY 3 - Maintain Connectivity Corridors:\n")
  text <- paste0(text, "  • Preserve stepping-stone reefs between populations\n")
  text <- paste0(text, "  • Focus on hub reefs that facilitate connectivity\n")
  
  # Priority 4: Seasonal management
  text <- paste0(text, "\nPRIORITY 4 - Seasonal Management:\n")
  spawning_names <- month.abb[spawning_months]
  text <- paste0(text, sprintf("  • Implement spawning protections (%s)\n",
                              paste(spawning_names, collapse = "-")))
  text <- paste0(text, "  • Time restoration outside peak spawning\n")
  
  # Priority 5: Monitoring
  text <- paste0(text, "\nPRIORITY 5 - Monitor Environmental Conditions:\n")
  text <- paste0(text, "  • Track temperature, salinity, pH at key sites\n")
  text <- paste0(text, "  • Establish early warning for stressors\n")
  
  return(text)
}

#' @title Create full analysis report
#' @param analysis_results List with all analysis results
#' @param output_file Output filename
#' @param include_plots Logical, whether to reference plots
#' @return Invisible NULL (writes to file)
create_full_report <- function(analysis_results, output_file,
                              include_plots = TRUE) {
  
  # Extract components
  reef_data <- analysis_results$reef_data
  env_summary <- analysis_results$env_summary
  monthly_data <- analysis_results$monthly_data
  conn_assessment <- analysis_results$conn_assessment
  metrics <- analysis_results$metrics
  config <- analysis_results$config
  
  # Start report
  report <- generate_header("OYSTER LARVAL DISPERSAL ANALYSIS - COMPREHENSIVE REPORT")
  
  # Executive summary
  report <- paste0(report, format_section("EXECUTIVE SUMMARY", 1))
  report <- paste0(report, sprintf("• Analyzed %d oyster reef sites\n", 
                                  nrow(reef_data)))
  report <- paste0(report, sprintf("• Total oyster density: %.2f\n", 
                                  sum(reef_data$AvgDensity)))
  report <- paste0(report, sprintf("• Mean connectivity: %.4f\n", 
                                  conn_assessment$mean_connectivity))
  
  # Environmental conditions
  report <- paste0(report, generate_env_summary(env_summary, monthly_data))
  
  # Connectivity patterns
  report <- paste0(report, generate_connectivity_summary(conn_assessment, metrics))
  
  # Key reefs
  report <- paste0(report, generate_key_reefs_summary(metrics))
  
  # Management recommendations
  report <- paste0(report, generate_recommendations(metrics))
  
  # Methods section
  report <- paste0(report, format_section("METHODS", 1))
  report <- paste0(report, "\nDispersal Model Parameters:\n")
  report <- paste0(report, sprintf("  • Pelagic larval duration: %d days\n",
                                  config$dispersal$pelagic_larval_duration))
  report <- paste0(report, sprintf("  • Daily mortality rate: %.0f%%\n",
                                  config$dispersal$mortality_rate * 100))
  report <- paste0(report, sprintf("  • Maximum dispersal: %d km\n",
                                  config$dispersal$max_dispersal_distance))
  
  # Output files
  if (include_plots) {
    report <- paste0(report, format_section("OUTPUT FILES", 1))
    report <- paste0(report, "\nFigures:\n")
    report <- paste0(report, "  • environmental_conditions.png\n")
    report <- paste0(report, "  • connectivity_matrix.png\n")
    report <- paste0(report, "  • network_classification.png\n")
    report <- paste0(report, "  • temporal_variations.png\n")
    report <- paste0(report, "  • settlement_probability.png\n")
    
    report <- paste0(report, "\nTables:\n")
    report <- paste0(report, "  • reef_metrics.csv\n")
    report <- paste0(report, "  • connectivity_matrix.csv\n")
    report <- paste0(report, "  • environmental_summary.csv\n")
  }
  
  # Footer
  report <- paste0(report, "\n", paste(rep("=", 80), collapse = ""), "\n")
  report <- paste0(report, "                         END OF REPORT\n")
  report <- paste0(report, paste(rep("=", 80), collapse = ""), "\n")
  
  # Write to file
  writeLines(report, output_file)
  message(paste("Report saved:", output_file))
  
  invisible(NULL)
}

#' @title Generate summary table
#' @param data Data frame to summarize
#' @param output_file CSV filename
#' @param digits Number of digits for rounding
#' @return Invisible data frame
export_summary_table <- function(data, output_file, digits = 3) {
  
  # Round numeric columns
  numeric_cols <- sapply(data, is.numeric)
  data[numeric_cols] <- round(data[numeric_cols], digits)
  
  # Write CSV
  write.csv(data, output_file, row.names = FALSE)
  message(paste("Table saved:", output_file))
  
  invisible(data)
}

#' @title Create markdown report
#' @param analysis_results Analysis results list
#' @param output_file Markdown filename
#' @return Invisible NULL
create_markdown_report <- function(analysis_results, output_file) {
  
  # This would create a markdown-formatted report
  # Implementation depends on specific needs
  
  message("Markdown report generation not yet implemented")
  invisible(NULL)
}