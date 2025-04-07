"""
Phyto-NAS-T Working Example
------------------------------
Demonstrates the correct usage pattern for the current implementation.
"""

import numpy as np
from phyto_nas_tsc import fit

# ---- Working Example ---- #
# This function demonstrates how to use the fit function with synthetic data
def working_example():
    print("\n=== Running Working Example ===")
    
    # 1. Creates properly formatted synthetic data
    np.random.seed(42)
    X = np.random.randn(150, 1, 50)     # 150 samples, 1 timestep, 50 features
    y = np.zeros((150, 2))              # one-hot encoded (2 classes)
    y[:75, 0] = 1                       # first half = class 0
    y[75:, 1] = 1                       # second half = class 1

    # 2. Runs optimization with all parameters in 'others'
    result = fit(
        X=X,
        y=y,
        others={
            'population_size': 4,       # must be in others dict
            'generations': 2,           # must be in others dict
            'timeout': 600,
            'early_stopping': True
        }
    )
    
    print("\n=== Results ===")
    print(f"Best Accuracy: {result['accuracy']:.4f}")
    print("Best Model Configuration:")
    for key, value in result['architecture'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    working_example()
    print("\nExample completed successfully!")