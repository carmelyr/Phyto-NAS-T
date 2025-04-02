"""Example usage of Phyto-NAS-TSC package"""
import numpy as np
from phyto_nas_tsc import fit

# Generate synthetic time series data
num_samples = 200
timesteps = 20
features = 1
num_classes = 3

X = np.random.randn(num_samples, timesteps, features)
y = np.eye(num_classes)[np.random.randint(0, num_classes, num_samples)]

# Run NAS optimization
print("Starting Neural Architecture Search...")
results = fit(
    X,
    y,
    scoring='accuracy',
    population_size=10,
    generations=3,
    verbose=True
)

print("\nOptimization complete!")
print(f"Best accuracy: {results['best_accuracy']:.4f}")
print("Best configuration:")
for k, v in results['best_config'].items():
    print(f"  {k}: {v}")