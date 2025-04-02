# Phyto-NAS-TSC

Neural Architecture Search for Time Series Classification

## Installation

```bash
pip install phyto-nas-tsc

## Quickstart

```python
import numpy as np
from phyto_nas_tsc import fit

# Synthetic data
X = np.random.rand(100, 10, 1)  # 100 samples, 10 timesteps, 1 feature
y = np.eye(2)[np.random.randint(0, 2, 100)]  # Binary classification

# Run optimization
results = fit(X, y, generations=3, population_size=5)
print(f"Best Architecture: {results['architecture']}")
```