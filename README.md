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

# Synthetic data
X = np.random.rand(100, 10, 1) 
y = np.eye(2)[np.random.randint(0, 2, 100)]

# Run optimization
results = fit(X, y, generations=3, population_size=5)
print(f"Best Architecture: {results['architecture']}")
```