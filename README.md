# Phyto-NAS-TSC

An evolutionary approach to automatically design optimal neural network architectures for time series classification tasks.

## Installation

```bash
pip install phyto-nas-tsc

## Installation directly from source
git clone https://github.com/carmelyr/Phyto-NAS-T.git
cd Phyto-NAS-T
pip install -e .

## Features

- Evolutionary algorithm for architecture search
- Optimized for time series data (1D signals)
- Optimized for LSTM model
- Tracks optimization history and metrics
- GPU-accelerated training

## Quickstart

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
    scoring='accuracy',                             # metric to optimize 
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