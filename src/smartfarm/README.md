# SmartFarm: Crop Growth Modeling and Optimization

A Python framework for simulating crop growth under varying environmental conditions and optimizing irrigation and fertilizer strategies using genetic algorithms.

## Motivation

Rising production costs and declining commodity prices have made farming increasingly expensive. This framework provides computational tools for optimizing resource application without requiring costly field trials. The core idea is that plants have "memory"—they don't respond instantly to water, fertilizer, temperature, and solar radiation, but absorb nutrients over time with characteristic delays. By modeling these delayed, cumulative dynamics, we can search for irrigation and fertilizer strategies that outperform conventional scheduling approaches.

- A **coupled ODE crop growth model** with five state variables (height, leaf area, leaf count, flower size, fruit biomass)
- **Delayed absorption** via FIR (Finite Impulse Response) convolution with Gaussian kernels
- **Cumulative stress tracking** via EMA (Exponential Moving Average) filtering
- **Genetic algorithm optimization** to find irrigation and fertilizer strategies that maximize net revenue

## Code Structure

```
src/smartfarm/
├── core/
│   ├── model/           # Crop growth model components
│   │   ├── model_params.py              # Time-stepping parameters (dt, simulation hours)
│   │   ├── model_carrying_capacities.py # Max values for each state variable
│   │   ├── model_growth_rates.py        # Baseline growth rates
│   │   ├── model_initial_conditions.py  # Starting values for state variables
│   │   ├── model_sensitivities.py       # FIR kernel sigmas, EMA betas
│   │   ├── model_typical_disturbances.py# Expected nutrient levels
│   │   ├── model_disturbances.py        # Environmental inputs (precip, temp, radiation)
│   │   └── model_helpers.py             # Logistic step, FIR kernels, utilities
│   │
│   ├── ga/              # Genetic algorithm components
│   │   ├── ga.py                # Main GA loop (selection, crossover, evolution)
│   │   ├── ga_member.py         # Individual solution: chromosome + cost evaluation
│   │   ├── ga_population.py     # Population management and batch evaluation
│   │   ├── ga_params.py         # GA hyperparameters and economic weights
│   │   ├── ga_bounds.py         # Search bounds for decision variables
│   │   └── ga_result.py         # Result storage and serialization
│   │
│   ├── plotting/        # Visualization utilities
│   │   └── plotting.py          # Functions for state trajectories, nutrient factors, etc.
│   │
│   └── aws/             # Cloud deployment (Lambda, EC2) for parallel evaluation
│
├── examples/            # Jupyter notebooks demonstrating usage
│
└── io/
    └── inputs/          # CSV files with weather data, precomputed values
```

## Example Notebooks

Start with these notebooks in order to understand the framework:

### 1. `examples/inputs_and_disturbances.ipynb`
**Start here.** Visualizes the environmental input data (precipitation, temperature, solar radiation) from historical records. Shows how raw hourly data is loaded and structured for simulation.

### 2. `examples/metabolism_analysis.ipynb`
**Understand the core modeling innovation.** Demonstrates how the FIR convolution and EMA filtering transform raw inputs into nutrient factors:
- Raw fertilizer application → Delayed absorption (via Gaussian FIR kernel)
- Cumulative absorption → Anomaly from expected levels
- Anomaly → EMA-smoothed divergence → Nutrient factor (0 to 1)

Shows how different `sigma` values (metabolic timescales) affect the delayed response curves. This notebook is essential for understanding why timing matters in this model.

### 3. `examples/ga/ga_single_simulation.ipynb`
**Run a single season simulation.** Given specific irrigation and fertilizer parameters (frequency and amount), simulates the full growing season and shows:
- State variable trajectories (h, A, N, c, P over time)
- Applied vs. absorbed nutrient curves
- Cumulative values vs. expected values
- Nutrient factor evolution
- Time-varying growth rates and carrying capacities
- Final profit/expense/revenue calculation

Use this to understand what the cost function "sees" when evaluating a candidate strategy.

### 4. `examples/ga/ga_analyze_results.ipynb`
**Analyze GA optimization results.** Loads saved GA runs and visualizes:
- Convergence curves (best cost and mean parent cost over generations)
- Final optimal strategy parameters
- Comparison of optimal vs. baseline strategies

### 5. `examples/ga/ga_sensitivity_analysis.ipynb`
**Explore parameter sensitivity.** Varies GA parameters or model parameters to understand robustness of solutions.

### 6. `examples/ga/ga_parallel.ipynb`
**Run GA with parallel evaluation.** Demonstrates how to use multiprocessing or AWS Lambda for faster population evaluation.

### 7. `examples/ga/ga_cpp.ipynb`
**Accelerated simulation.** Uses a C++ implementation of the cost function for faster evaluation (useful for large populations or many generations).

## Quick Start

### Running a Single Simulation

```python
import numpy as np
import pandas as pd

from core.ga.ga_member import Member
from core.ga.ga_params import GeneticAlgorithmParams
from core.model.model_params import ModelParams
from core.model.model_carrying_capacities import ModelCarryingCapacities
from core.model.model_growth_rates import ModelGrowthRates
from core.model.model_initial_conditions import ModelInitialConditions
from core.model.model_sensitivities import ModelSensitivities
from core.model.model_typical_disturbances import ModelTypicalDisturbances
from core.model.model_disturbances import ModelDisturbances

# Load weather data
weather = pd.read_csv('io/inputs/hourly_prcp_rad_temp_iowa.csv')
disturbances = ModelDisturbances(
    precipitation = weather['Hourly Precipitation (in)'].to_numpy(),
    radiation     = weather['Hourly Radiation (W/m2)'].to_numpy(),
    temperature   = weather['Temperature (C)'].to_numpy()
)

# Configure model (using corn defaults)
model_params = ModelParams(dt=0.1, simulation_hours=2900, closed_form=True)
carrying_capacities = ModelCarryingCapacities()  # defaults for corn
growth_rates = ModelGrowthRates()
sensitivities = ModelSensitivities()
initial_conditions = ModelInitialConditions(
    h0=carrying_capacities.kh / model_params.simulation_hours,
    A0=carrying_capacities.kA / model_params.simulation_hours,
    N0=carrying_capacities.kN / model_params.simulation_hours,
    c0=carrying_capacities.kc / model_params.simulation_hours,
    P0=carrying_capacities.kP / model_params.simulation_hours
)
typical_disturbances = ModelTypicalDisturbances()

# Define an irrigation/fertilizer strategy
strategy = np.array([
    336,   # irrigation frequency (hours) - every 14 days
    0.23,  # irrigation amount (inches)
    2160,  # fertilizer frequency (hours) - every 90 days
    175    # fertilizer amount (lbs)
])

# Evaluate the strategy
member = Member(
    ga_params=GeneticAlgorithmParams(),
    carrying_capacities=carrying_capacities,
    disturbances=disturbances,
    growth_rates=growth_rates,
    initial_conditions=initial_conditions,
    model_params=model_params,
    typical_disturbances=typical_disturbances,
    sensitivities=sensitivities,
    values=strategy
)

cost = member.get_cost()
print(f"Net revenue: ${-cost:.2f}")  # Cost is negative revenue
```

### Running the Genetic Algorithm

```python
from core.ga.ga import GeneticAlgorithm
from core.ga.ga_bounds import GABounds

# Define search bounds
bounds = GABounds(
    lower_bounds=np.array([100, 0.5, 700, 100]),   # [irr_freq, irr_amt, fert_freq, fert_amt]
    upper_bounds=np.array([700, 5.0, 2900, 500])
)

# Run optimization
ga = GeneticAlgorithm(
    ga_params=GeneticAlgorithmParams(
        population_size=128,
        num_parents=16,
        num_children=16,
        num_generations=100
    ),
    bounds=bounds,
    carrying_capacities=carrying_capacities,
    disturbances=disturbances,
    growth_rates=growth_rates,
    initial_conditions=initial_conditions,
    model_params=model_params,
    typical_disturbances=typical_disturbances,
    sensitivities=sensitivities
)

result = ga.run()
print(f"Best strategy: {result.best_member.values}")
print(f"Best revenue: ${-result.best_cost:.2f}")
```

## Key Parameters

### Model Parameters (`ModelParams`)
- `dt`: Time step in hours (default: 0.1)
- `simulation_hours`: Growing season length (default: 2900 for corn, ~121 days)
- `closed_form`: Use analytical logistic solution vs. Euler integration

### Sensitivities (`ModelSensitivities`)
- `sigma_W`, `sigma_F`, `sigma_T`, `sigma_R`: FIR kernel spreads (hours) for water, fertilizer, temperature, radiation
- `beta_divergence`: EMA parameter for cumulative anomaly tracking
- `beta_nutrient_factor`: EMA parameter for nutrient factor smoothing
- `alpha`: Exponential decay rate for converting divergence to stress

### GA Parameters (`GeneticAlgorithmParams`)
- `population_size`: Number of candidate solutions per generation
- `num_parents`: Elite solutions kept each generation
- `num_children`: New solutions created via crossover
- `num_generations`: Total generations to run
- `weight_*`: Economic weights for profit/cost calculation

## Data Requirements

The model requires hourly environmental data:
- **Precipitation** (inches): From NOAA Climate Data Online
- **Temperature** (°C): From NSRDB (National Solar Radiation Database)
- **Solar Radiation** (W/m²): From NSRDB

Example data for Fairfax, Iowa is provided in `io/inputs/hourly_prcp_rad_temp_iowa.csv`.

## References

This codebase accompanies the paper:

> Becker, C.J. and Zohdi, T.I. "Optimizing irrigation and fertilizer strategy using a crop growth model with delayed absorption dynamics." *Computers and Electronics in Agriculture* (in preparation).

See the `paper/` directory for the manuscript and bibliography.
