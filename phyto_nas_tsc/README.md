
### 2. `phyto_nas_tsc/README.md` (Package Documentation)

```markdown
# Phyto-NAS-TSC Package Documentation

## File Overview

### Core Modules

1. **`core.py`**
   - Main API endpoint (`fit()` function)
   - Handles data loading and validation
   - Coordinates optimization process

2. **`_config.py`**
   - Hyperparameter defaults
   - Architecture search space definition
   - Device configuration (CPU/GPU)

3. **`_data_handler.py`**
   - Data loading and preprocessing
   - Input validation
   - Dataset splitting

### Optimization Components

4. **`_evolutionary_algorithm.py`**
   - NASDifferentialEvolution class
   - Population management
   - Mutation and crossover operations
   - Fitness evaluation

5. **`_optimizer.py`**
   - NASOptimizer wrapper class
   - Progress tracking
   - Result aggregation

### Model Components

6. **`_model_builder.py`**
   - Neural network construction
   - Supported layer types:
     - LSTM, CNN, Dense, Attention
   - Model training logic

7. **`_utils.py`**
   - Fitness function calculation
   - Result saving (CSV/JSON)
   - Helper functions

## Development Guide

### Adding New Layer Types

1. Modify `_config.py`:
   - Add to `layer_options`
   - Define parameter ranges

2. Update `_model_builder.py`:
   - Implement new layer class
   - Add to `build_model()` function

Example:
```python
# In _config.py
layer_options.append({
    'layer': 'Transformer',
    'heads': [2, 4],
    'ff_dim': [64, 128]
})