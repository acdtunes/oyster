################################################################################
# Statistical Analysis Functions
# Statistical tests, correlations, and model fitting
################################################################################

#' @title Perform correlation analysis
#' @param data Data frame with variables
#' @param variables Character vector of variables to correlate
#' @param method Correlation method: "pearson", "spearman", "kendall"
#' @return Correlation matrix with significance
correlation_analysis <- function(data, variables = NULL, method = "pearson") {
  
  # Select variables
  if (is.null(variables)) {
    # Use all numeric columns
    variables <- names(data)[sapply(data, is.numeric)]
  }
  
  # Extract data
  cor_data <- data[, variables, drop = FALSE]
  
  # Calculate correlation matrix
  cor_matrix <- cor(cor_data, use = "complete.obs", method = method)
  
  # Calculate p-values
  n <- nrow(cor_data)
  p_matrix <- matrix(NA, ncol(cor_matrix), nrow(cor_matrix))
  
  for (i in 1:ncol(cor_matrix)) {
    for (j in 1:nrow(cor_matrix)) {
      if (i != j) {
        test <- cor.test(cor_data[, i], cor_data[, j], method = method)
        p_matrix[i, j] <- test$p.value
      }
    }
  }
  
  dimnames(p_matrix) <- dimnames(cor_matrix)
  
  return(list(
    correlation = cor_matrix,
    p_values = p_matrix,
    n = n,
    method = method
  ))
}

#' @title Fit connectivity model
#' @param metrics Network metrics data frame
#' @param predictors Character vector of predictor variables
#' @param response Response variable name
#' @return Linear model object
fit_connectivity_model <- function(metrics, 
                                  predictors = c("Density", "Temperature", 
                                               "Salinity", "CurrentSpeed"),
                                  response = "OutStrength") {
  
  # Build formula
  formula_str <- paste(response, "~", paste(predictors, collapse = " + "))
  formula_obj <- as.formula(formula_str)
  
  # Fit model
  model <- lm(formula_obj, data = metrics)
  
  return(model)
}

#' @title Perform ANOVA on environmental conditions
#' @param data Data frame with groups and values
#' @param group_var Grouping variable name
#' @param value_var Value variable name
#' @return ANOVA results
perform_anova <- function(data, group_var, value_var) {
  
  # Build formula
  formula_obj <- as.formula(paste(value_var, "~", group_var))
  
  # Perform ANOVA
  aov_result <- aov(formula_obj, data = data)
  
  # Get summary
  summary_aov <- summary(aov_result)
  
  # Post-hoc test if significant
  if (summary_aov[[1]][["Pr(>F)"]][1] < 0.05) {
    posthoc <- TukeyHSD(aov_result)
  } else {
    posthoc <- NULL
  }
  
  return(list(
    anova = aov_result,
    summary = summary_aov,
    posthoc = posthoc
  ))
}

#' @title Calculate summary statistics by group
#' @param data Data frame
#' @param group_var Grouping variable
#' @param value_vars Value variables to summarize
#' @return Summary statistics data frame
group_summary_stats <- function(data, group_var, value_vars) {
  
  results <- list()
  
  for (var in value_vars) {
    summary_df <- data %>%
      group_by(!!sym(group_var)) %>%
      summarise(
        n = n(),
        mean = mean(!!sym(var), na.rm = TRUE),
        sd = sd(!!sym(var), na.rm = TRUE),
        se = sd(!!sym(var), na.rm = TRUE) / sqrt(n()),
        median = median(!!sym(var), na.rm = TRUE),
        min = min(!!sym(var), na.rm = TRUE),
        max = max(!!sym(var), na.rm = TRUE),
        .groups = "drop"
      ) %>%
      mutate(variable = var)
    
    results[[var]] <- summary_df
  }
  
  combined <- do.call(rbind, results)
  return(combined)
}

#' @title Test for spatial autocorrelation
#' @param values Vector of values
#' @param coords Matrix or data frame with x, y coordinates
#' @param method "moran" or "geary"
#' @return Spatial autocorrelation test results
test_spatial_autocorrelation <- function(values, coords, method = "moran") {
  
  # Calculate distance matrix
  dist_matrix <- as.matrix(dist(coords))
  
  # Create weights matrix (inverse distance)
  weights <- 1 / dist_matrix
  diag(weights) <- 0
  weights <- weights / rowSums(weights)
  
  # Calculate Moran's I or Geary's C
  n <- length(values)
  
  if (method == "moran") {
    # Moran's I
    mean_val <- mean(values, na.rm = TRUE)
    numerator <- sum(weights * outer(values - mean_val, values - mean_val))
    denominator <- sum((values - mean_val)^2)
    I <- (n / sum(weights)) * (numerator / denominator)
    
    # Expected value and variance under null hypothesis
    E_I <- -1 / (n - 1)
    
    result <- list(
      statistic = I,
      expected = E_I,
      method = "Moran's I"
    )
  } else if (method == "geary") {
    # Geary's C
    numerator <- sum(weights * outer(values, values, FUN = function(x, y) (x - y)^2))
    denominator <- sum((values - mean(values))^2)
    C <- ((n - 1) / (2 * sum(weights))) * (numerator / denominator)
    
    result <- list(
      statistic = C,
      expected = 1,  # Expected value under null hypothesis
      method = "Geary's C"
    )
  } else {
    stop("Method must be 'moran' or 'geary'")
  }
  
  return(result)
}

#' @title Compare connectivity between seasons
#' @param conn_matrix_spawn Spawning season connectivity matrix
#' @param conn_matrix_nonspawn Non-spawning season connectivity matrix
#' @return Comparison results
compare_seasonal_connectivity <- function(conn_matrix_spawn, conn_matrix_nonspawn) {
  
  # Flatten matrices (excluding diagonal)
  spawn_values <- conn_matrix_spawn[upper.tri(conn_matrix_spawn)]
  nonspawn_values <- conn_matrix_nonspawn[upper.tri(conn_matrix_nonspawn)]
  
  # Paired t-test (if same reef pairs)
  if (length(spawn_values) == length(nonspawn_values)) {
    t_test <- t.test(spawn_values, nonspawn_values, paired = TRUE)
  } else {
    t_test <- t.test(spawn_values, nonspawn_values, paired = FALSE)
  }
  
  # Wilcoxon test (non-parametric alternative)
  wilcox_test <- wilcox.test(spawn_values, nonspawn_values, paired = TRUE)
  
  # Summary statistics
  summary_stats <- data.frame(
    Season = c("Spawning", "Non-spawning"),
    Mean = c(mean(spawn_values), mean(nonspawn_values)),
    SD = c(sd(spawn_values), sd(nonspawn_values)),
    Median = c(median(spawn_values), median(nonspawn_values)),
    Max = c(max(spawn_values), max(nonspawn_values))
  )
  
  return(list(
    t_test = t_test,
    wilcox_test = wilcox_test,
    summary = summary_stats
  ))
}

#' @title Calculate diversity indices for reef network
#' @param conn_matrix Connectivity matrix
#' @return List of diversity indices
calculate_diversity_indices <- function(conn_matrix) {
  
  # Shannon diversity of connections
  p <- conn_matrix / sum(conn_matrix)
  p <- p[p > 0]
  shannon <- -sum(p * log(p))
  
  # Simpson diversity
  simpson <- 1 - sum(p^2)
  
  # Evenness
  n <- length(p)
  evenness <- shannon / log(n)
  
  # Connectivity concentration (Gini coefficient)
  values <- sort(as.vector(conn_matrix))
  n <- length(values)
  index <- 1:n
  gini <- 2 * sum(index * values) / (n * sum(values)) - (n + 1) / n
  
  return(list(
    shannon = shannon,
    simpson = simpson,
    evenness = evenness,
    gini = gini
  ))
}

#' @title Bootstrap confidence intervals for connectivity
#' @param reef_data Reef data
#' @param nc_data NetCDF data
#' @param params Dispersal parameters
#' @param n_boot Number of bootstrap iterations
#' @param conf_level Confidence level
#' @return Bootstrap results
bootstrap_connectivity <- function(reef_data, nc_data, params, 
                                  n_boot = 100, conf_level = 0.95) {
  
  n_reefs <- nrow(reef_data)
  boot_matrices <- array(NA, dim = c(n_reefs, n_reefs, n_boot))
  
  # Progress bar
  pb <- txtProgressBar(min = 0, max = n_boot, style = 3)
  
  for (b in 1:n_boot) {
    # Resample reefs with replacement
    sample_idx <- sample(1:n_reefs, n_reefs, replace = TRUE)
    reef_sample <- reef_data[sample_idx, ]
    
    # Calculate connectivity matrix
    conn_matrix <- build_connectivity_matrix(reef_sample, nc_data, params, 
                                            verbose = FALSE)
    boot_matrices[, , b] <- conn_matrix
    
    setTxtProgressBar(pb, b)
  }
  close(pb)
  
  # Calculate confidence intervals
  alpha <- (1 - conf_level) / 2
  lower_quantile <- apply(boot_matrices, c(1, 2), quantile, probs = alpha, na.rm = TRUE)
  upper_quantile <- apply(boot_matrices, c(1, 2), quantile, probs = 1 - alpha, na.rm = TRUE)
  mean_matrix <- apply(boot_matrices, c(1, 2), mean, na.rm = TRUE)
  
  return(list(
    mean = mean_matrix,
    lower = lower_quantile,
    upper = upper_quantile,
    conf_level = conf_level,
    n_boot = n_boot
  ))
}

#' @title Model selection for environmental predictors
#' @param data Data frame with response and predictors
#' @param response Response variable name
#' @param predictors Character vector of predictor names
#' @return Model selection results
model_selection <- function(data, response, predictors) {
  
  # Generate all possible models
  n_pred <- length(predictors)
  models <- list()
  aic_values <- numeric()
  
  # Null model
  null_formula <- as.formula(paste(response, "~ 1"))
  null_model <- lm(null_formula, data = data)
  models[["null"]] <- null_model
  aic_values["null"] <- AIC(null_model)
  
  # All combinations of predictors
  for (i in 1:n_pred) {
    combos <- combn(predictors, i)
    
    for (j in 1:ncol(combos)) {
      pred_set <- combos[, j]
      formula_str <- paste(response, "~", paste(pred_set, collapse = " + "))
      formula_obj <- as.formula(formula_str)
      
      model <- lm(formula_obj, data = data)
      model_name <- paste(pred_set, collapse = "_")
      
      models[[model_name]] <- model
      aic_values[model_name] <- AIC(model)
    }
  }
  
  # Sort by AIC
  aic_df <- data.frame(
    Model = names(aic_values),
    AIC = aic_values,
    Delta_AIC = aic_values - min(aic_values),
    stringsAsFactors = FALSE
  ) %>%
    arrange(AIC)
  
  # Best model
  best_model <- models[[aic_df$Model[1]]]
  
  return(list(
    best_model = best_model,
    aic_table = aic_df,
    all_models = models
  ))
}