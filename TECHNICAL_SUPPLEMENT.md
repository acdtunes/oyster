# Technical Supplement: Advanced Implementation and Future Directions

## Advanced Model Implementation

### 1. Particle-Based Lagrangian Model

Transform the current Eulerian approach to Individual-Based Model (IBM):

```r
# Particle tracking implementation
track_particles <- function(release_location, n_particles, nc_data, params) {
  
  # Initialize particle positions
  particles <- data.frame(
    id = 1:n_particles,
    lon = rnorm(n_particles, release_location$lon, 0.001),
    lat = rnorm(n_particles, release_location$lat, 0.001),
    age = 0,
    alive = TRUE,
    settled = FALSE
  )
  
  # Time loop
  for (day in 1:params$pelagic_larval_duration) {
    
    # Extract currents at particle locations
    u_interp <- bilinear_interpolate(nc_data$u_surface, particles$lon, particles$lat, day)
    v_interp <- bilinear_interpolate(nc_data$v_surface, particles$lon, particles$lat, day)
    
    # Advection (4th-order Runge-Kutta)
    k1_lon <- u_interp * dt / meters_per_degree_lon
    k1_lat <- v_interp * dt / meters_per_degree_lat
    
    k2_lon <- bilinear_interpolate(nc_data$u_surface, 
                                   particles$lon + k1_lon/2, 
                                   particles$lat + k1_lat/2, day) * dt / meters_per_degree_lon
    # ... continue RK4
    
    # Diffusion (random walk)
    particles$lon <- particles$lon + rnorm(n_particles, 0, sqrt(2 * params$diffusion * dt))
    particles$lat <- particles$lat + rnorm(n_particles, 0, sqrt(2 * params$diffusion * dt))
    
    # Vertical migration (diel)
    hour <- (day * 24) %% 24
    if (hour >= 6 && hour <= 18) {
      particles$depth <- params$daytime_depth
    } else {
      particles$depth <- params$nighttime_depth
    }
    
    # Mortality (stochastic)
    particles$alive <- particles$alive & (runif(n_particles) > params$mortality_rate)
    
    # Settlement (if competent and near suitable habitat)
    if (day >= params$settlement_competency) {
      habitat_quality <- extract_habitat(particles$lon, particles$lat)
      settlement_prob <- params$settlement_rate * habitat_quality
      particles$settled <- particles$settled | (runif(n_particles) < settlement_prob)
    }
    
    # Update age
    particles$age <- particles$age + 1
  }
  
  return(particles)
}
```

### 2. Machine Learning Enhancement

Use neural networks to predict connectivity from environmental features:

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Build neural network for connectivity prediction
def build_connectivity_model(input_dim):
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(input_dim,)),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')  # Connectivity probability
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy', 'AUC']
    )
    
    return model

# Feature engineering
def extract_features(source_reef, sink_reef, environmental_data):
    features = []
    
    # Distance features
    features.append(haversine_distance(source_reef, sink_reef))
    features.append(bearing(source_reef, sink_reef))
    
    # Current features
    features.append(mean_current_speed(source_reef))
    features.append(current_direction(source_reef))
    features.append(current_toward_sink(source_reef, sink_reef))
    
    # Environmental features
    features.append(temperature_difference(source_reef, sink_reef))
    features.append(salinity_difference(source_reef, sink_reef))
    features.append(depth_difference(source_reef, sink_reef))
    
    # Temporal features
    features.append(spawning_season_overlap(source_reef, sink_reef))
    features.append(tidal_phase_alignment(source_reef, sink_reef))
    
    return np.array(features)

# Train model on observed connectivity data
X_train = np.array([extract_features(s, t, env_data) 
                    for s, t in training_pairs])
y_train = observed_connectivity[training_pairs]

model = build_connectivity_model(X_train.shape[1])
history = model.fit(X_train, y_train, 
                   epochs=100, 
                   validation_split=0.2,
                   callbacks=[
                       keras.callbacks.EarlyStopping(patience=10),
                       keras.callbacks.ReduceLROnPlateau(patience=5)
                   ])
```

### 3. Optimization for Restoration Planning

Multi-objective optimization for site selection:

```r
library(GA)
library(mco)

# Define objectives
objectives <- function(x, reef_data, conn_matrix, costs) {
  selected <- which(x == 1)
  
  if (length(selected) == 0) {
    return(c(-Inf, -Inf, Inf))
  }
  
  # Objective 1: Maximize total larval production
  total_production <- sum(reef_data$density[selected] * reef_data$area[selected])
  
  # Objective 2: Maximize network connectivity
  subnet_conn <- conn_matrix[selected, selected]
  network_strength <- sum(subnet_conn)
  
  # Objective 3: Minimize cost
  total_cost <- sum(costs[selected])
  
  # Objective 4: Maximize resilience (eigenvalue)
  if (length(selected) > 1) {
    resilience <- max(Re(eigen(subnet_conn)$values))
  } else {
    resilience <- 0
  }
  
  return(c(total_production, network_strength, -total_cost, resilience))
}

# Run NSGA-II multi-objective optimization
nsga2_result <- nsga2(
  fn = objectives,
  idim = nrow(reef_data),
  odim = 4,
  lower.bounds = rep(0, nrow(reef_data)),
  upper.bounds = rep(1, nrow(reef_data)),
  popsize = 100,
  generations = 500,
  cprob = 0.7,
  mprob = 0.1,
  reef_data = reef_data,
  conn_matrix = conn_matrix,
  costs = restoration_costs
)

# Extract Pareto front
pareto_solutions <- nsga2_result$value
pareto_sites <- nsga2_result$par
```

### 4. Real-Time Forecasting System

Operational connectivity forecasting using streaming data:

```python
import xarray as xr
from datetime import datetime, timedelta
import asyncio

class ConnectivityForecaster:
    def __init__(self, model_config):
        self.config = model_config
        self.current_forecast = None
        self.particle_cloud = None
        
    async def update_hydrodynamics(self):
        """Fetch latest ROMS/HYCOM forecast"""
        async with aiohttp.ClientSession() as session:
            latest_forecast = await fetch_ocean_forecast(session)
            self.process_forecast(latest_forecast)
    
    def process_forecast(self, forecast_data):
        """Process 72-hour forecast into connectivity prediction"""
        # Extract relevant variables
        u_forecast = forecast_data['u_surface']
        v_forecast = forecast_data['v_surface']
        temp_forecast = forecast_data['temperature']
        
        # Run particle tracking for each spawning event
        connectivity_forecast = {}
        
        for reef in self.config['reefs']:
            # Estimate spawning probability
            spawn_prob = self.spawning_model(temp_forecast, reef)
            
            if spawn_prob > self.config['spawn_threshold']:
                # Release virtual particles
                particles = self.release_particles(
                    reef, 
                    n_particles=int(spawn_prob * 10000)
                )
                
                # Track for PLD
                trajectories = self.track_particles(
                    particles, 
                    u_forecast, 
                    v_forecast,
                    duration=self.config['pld']
                )
                
                # Calculate settlement
                connectivity_forecast[reef] = self.calculate_settlement(
                    trajectories,
                    self.config['reefs']
                )
        
        self.current_forecast = connectivity_forecast
        self.last_update = datetime.now()
    
    def spawning_model(self, temperature, reef):
        """Predict spawning probability from temperature"""
        optimal_temp = 25.0  # °C
        temp_at_reef = temperature.sel(
            lon=reef['lon'], 
            lat=reef['lat'],
            method='nearest'
        ).mean()
        
        # Gaussian spawning response
        spawn_prob = np.exp(-0.5 * ((temp_at_reef - optimal_temp) / 2.0) ** 2)
        
        # Lunar phase correction
        lunar_phase = self.get_lunar_phase()
        if lunar_phase in ['full', 'new']:
            spawn_prob *= 1.5
            
        return min(spawn_prob, 1.0)
    
    async def run_forecast_loop(self):
        """Continuous forecasting loop"""
        while True:
            try:
                await self.update_hydrodynamics()
                self.broadcast_forecast()
                await asyncio.sleep(3600)  # Update hourly
            except Exception as e:
                self.log_error(e)
                await asyncio.sleep(300)  # Retry in 5 minutes
```

### 5. Genetic Algorithm for Network Design

Optimize reef placement for maximum connectivity:

```r
library(igraph)

design_optimal_network <- function(potential_sites, target_connectivity, budget) {
  
  # Genetic algorithm fitness function
  fitness <- function(chromosome) {
    selected_sites <- potential_sites[chromosome == 1, ]
    
    if (nrow(selected_sites) == 0) return(0)
    
    # Build connectivity matrix for selected sites
    conn <- build_connectivity_matrix(selected_sites)
    
    # Calculate network metrics
    g <- graph_from_adjacency_matrix(conn, weighted = TRUE)
    
    # Metrics to optimize
    metrics <- list(
      connectivity = mean(conn[conn > 0]),
      clustering = transitivity(g),
      resilience = eigen_centrality(g)$value,
      cost = sum(selected_sites$restoration_cost)
    )
    
    # Composite fitness (weighted sum)
    fitness_score <- 
      metrics$connectivity * 0.3 +
      metrics$clustering * 0.2 +
      metrics$resilience * 0.3 -
      metrics$cost / budget * 0.2
    
    # Penalty for exceeding budget
    if (metrics$cost > budget) {
      fitness_score <- fitness_score * 0.1
    }
    
    return(fitness_score)
  }
  
  # Run genetic algorithm
  ga_result <- ga(
    type = "binary",
    fitness = fitness,
    nBits = nrow(potential_sites),
    maxiter = 1000,
    popSize = 200,
    pmutation = 0.05,
    pcrossover = 0.8,
    elitism = 0.1 * 200
  )
  
  # Extract optimal design
  optimal_design <- potential_sites[ga_result@solution[1,] == 1, ]
  
  return(list(
    sites = optimal_design,
    fitness = ga_result@fitnessValue,
    generations = ga_result@iter
  ))
}
```

### 6. Climate Change Projections

Project future connectivity under climate scenarios:

```r
project_future_connectivity <- function(base_connectivity, climate_scenarios) {
  
  projections <- list()
  
  for (scenario in names(climate_scenarios)) {
    
    # Extract climate projections
    temp_change <- climate_scenarios[[scenario]]$temperature_change
    current_change <- climate_scenarios[[scenario]]$current_change
    sea_level_rise <- climate_scenarios[[scenario]]$slr
    
    # Adjust PLD for temperature
    future_pld <- base_pld * exp(-0.05 * temp_change)  # Q10 = 2
    
    # Adjust mortality for temperature stress
    if (temp_change > 2) {
      future_mortality <- base_mortality * 1.5
    } else {
      future_mortality <- base_mortality
    }
    
    # Adjust currents
    future_currents <- base_currents * (1 + current_change)
    
    # Recalculate connectivity
    future_conn <- build_connectivity_matrix(
      reef_data = adjust_reef_depths(reef_data, sea_level_rise),
      currents = future_currents,
      params = list(
        pld = future_pld,
        mortality = future_mortality,
        diffusion = base_diffusion * (1 + temp_change * 0.1)
      )
    )
    
    # Calculate change metrics
    projections[[scenario]] <- list(
      connectivity_matrix = future_conn,
      mean_change = mean(future_conn) / mean(base_connectivity) - 1,
      resilience_change = calculate_resilience(future_conn) / 
                         calculate_resilience(base_connectivity) - 1,
      n_isolated = sum(rowSums(future_conn) < 0.01)
    )
  }
  
  return(projections)
}
```

## Performance Optimization

### Parallel Processing
```r
library(future)
library(furrr)
plan(multisession, workers = parallel::detectCores() - 1)

# Parallel connectivity calculation
future_connectivity <- future_map2_dfr(
  source_reefs,
  sink_reefs,
  calculate_connectivity,
  .options = furrr_options(seed = 123)
)
```

### GPU Acceleration
```python
import cupy as cp
import cupyx.scipy.sparse as sp

def gpu_dispersal_kernel(distance_matrix, diffusion_coeff, pld):
    """Calculate dispersal kernel on GPU"""
    d_gpu = cp.asarray(distance_matrix)
    sigma = cp.sqrt(2 * diffusion_coeff * pld * 86400) / 1000
    
    # Gaussian kernel
    kernel = cp.exp(-d_gpu**2 / (2 * sigma**2))
    
    # Apply mortality
    survival = (1 - 0.1)**pld
    kernel *= survival
    
    return cp.asnumpy(kernel)
```

## Data Management

### Database Schema
```sql
-- PostgreSQL schema for connectivity database
CREATE TABLE reefs (
    reef_id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    location GEOGRAPHY(POINT, 4326),
    density REAL,
    area REAL,
    metadata JSONB
);

CREATE TABLE connectivity (
    source_id INTEGER REFERENCES reefs(reef_id),
    sink_id INTEGER REFERENCES reefs(reef_id),
    year INTEGER,
    month INTEGER,
    probability REAL,
    particles_released INTEGER,
    particles_settled INTEGER,
    PRIMARY KEY (source_id, sink_id, year, month)
);

CREATE TABLE environmental_conditions (
    reef_id INTEGER REFERENCES reefs(reef_id),
    timestamp TIMESTAMP,
    temperature REAL,
    salinity REAL,
    current_u REAL,
    current_v REAL,
    ph REAL,
    oxygen REAL
);

-- Spatial index for efficient queries
CREATE INDEX idx_reef_location ON reefs USING GIST(location);
```

## Validation Framework

### Cross-validation with Field Data
```r
validate_model <- function(model_predictions, field_observations) {
  
  # Parentage analysis validation
  genetic_connectivity <- calculate_genetic_connectivity(
    parent_genotypes,
    offspring_genotypes,
    reef_locations
  )
  
  # Settlement plate validation
  observed_settlement <- aggregate_settlement_data(
    settlement_plates,
    time_period
  )
  
  # Metrics
  validation_metrics <- list(
    correlation = cor(model_predictions, field_observations),
    rmse = sqrt(mean((model_predictions - field_observations)^2)),
    bias = mean(model_predictions - field_observations),
    skill_score = 1 - var(model_predictions - field_observations) / 
                      var(field_observations)
  )
  
  # Bootstrap confidence intervals
  boot_metrics <- boot(
    data = cbind(model_predictions, field_observations),
    statistic = function(d, i) {
      cor(d[i,1], d[i,2])
    },
    R = 1000
  )
  
  validation_metrics$ci <- boot.ci(boot_metrics, type = "perc")
  
  return(validation_metrics)
}
```

## Future Research Directions

### Priority 1: Behavioral Complexity
- Ontogenetic vertical migration
- Chemotaxis toward settlement cues
- Predator avoidance behaviors
- Turbulence-mediated aggregation

### Priority 2: Environmental Stochasticity
- Episodic events (storms, freshets)
- Interannual variability (ENSO, NAO)
- Extreme temperature events
- Hypoxia effects on survival

### Priority 3: Ecological Interactions
- Predation pressure mapping
- Food limitation effects
- Competition for settlement space
- Disease transmission pathways

### Priority 4: Socioeconomic Integration
- Cost-benefit optimization
- Stakeholder preference modeling
- Ecosystem service valuation
- Adaptive management frameworks

---

*Technical Supplement Version 1.0*  
*Last Updated: November 2024*