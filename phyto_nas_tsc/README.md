# Phyto-NAS-TSC Package Documentation

## Overview

Phyto-NAS-TSC is a Python package for Neural Architecture Search (NAS) focused on Time Series Classification (TSC) tasks, which is specifically designed for plant electrophysiological data analysis.

The package implements a differential evolution algorithm to automatically discover optimal LSTM-based architectures.

## File Structure

### Core Modules

1. **`core.py`**
   - Main API endpoint (`fit()` function)
   - Handles data loading and validation
   - Coordinates the optimization process
   - Parameters:
     - X: Input features (n_samples, timesteps, features)
     - y: One-hot encoded labels
     - data_dir: Path to data directory
     - others: Additional optimization parameters

2. **`_config.py`**
   - Hyperparameter defaults:
     - population_size: Number of individuals in population
     - generations: Number of evolutionary generations
     - mutation/crossover rates
     - Early stopping and timeout configurations
   - Device configuration (CPU/GPU)

### Data Handling

3. **`_data_handler.py`**
   - Data loading and preprocessing:
     - Handles missing values (mean imputation)
     - Robust scaling (RobustScaler)
     - One-hot encoding for labels
     - Reshapes data for LSTM input (3D tensors)
   - Input validation (`validate_inputs()`)
   - Dataset splitting (5-fold cross-validation)

### Optimization Components

4. **`_evolutionary_algorithm.py`**
   - NASDifferentialEvolution class:
     - Population initialization
     - Mutation and crossover operations
     - Hybrid mutation factor calculation (exponential + linear decay)
     - Model evaluation with cross-validation
     - Fitness-based selection
   - Supports:
     - Early stopping
     - Timeout handling
     - Result logging

5. **`_optimizer.py`**
   - NASOptimizer wrapper class
   - Coordinates the evolutionary process
   - Handles result aggregation

### Model Components

6. **`_model_builder.py`**
   - LSTM model implementation (PyTorch Lightning)
   - Features:
     - Bidirectional processing
     - Optional attention mechanism
     - Layer normalization
     - Configurable dropout
   - Training configuration:
     - AdamW optimizer
     - Cyclical learning rate
     - L2 regularization

7. **`_utils.py`**
   - Fitness function calculation:
     - Balances accuracy, model size, and training time
   - Result saving to CSV
   - Helper functions

## Usage Example

```python
import numpy as np
from phyto_nas_tsc import fit
# from importlib.resources import files             # uncomment to use built-in data

# OPTION 1: Use your own data
X = np.random.randn(100, 1, 10)                     # 100 samples, 1 timestep, 10 features
y = np.zeros((100, 2))                              # one-hot encoded labels
y[:50, 0] = 1                                       # first 50 samples = class 0
y[50:, 1] = 1                                       # next 50 samples = class 1

# OPTION 2: Use built-in dataset (uncomment below)
# data_dir = str(files('phyto_nas_tsc.data'))       # path to included data
# X, y = None, None                                 # let the package load data automatically

# Run optimization
result = fit(
    X=X,                                            # comment out if using built-in data
    y=y,                                            # comment out if using built-in data
    # data_dir=data_dir,                            # uncomment if using built-in data
    others={
        'population_size': 5,    # required
        'generations': 3,        # required
        'early_stopping': True   # optional
    }
)

print(f"Best Accuracy: {result['accuracy']:.4f}")
print("Best Architecture:")
for param, value in result['architecture'].items():
    print(f"  {param}: {value}")
```