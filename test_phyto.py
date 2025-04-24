import numpy as np
from phyto_nas_tsc import fit

# OPTION 1: Use your own data (uncomment)
#X = np.random.randn(100, 1, 10)                     # 100 samples, 1 timestep, 10 features
#y = np.zeros((100, 2))                              # one-hot encoded labels
#y[:50, 0] = 1                                       # first 50 samples = class 0
#y[50:, 1] = 1                                       # next 50 samples = class 1

# OPTION 2: Use built-in dataset                     # let the package load data automatically
# Run optimization
result = fit(
    X=None,                                          # X is None, so the package will load data
    y=None,                                
    data_dir=None,                      
    scoring='accuracy',                                
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