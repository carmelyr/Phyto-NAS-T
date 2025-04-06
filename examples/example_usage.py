"""
Phyto-NAS-T Example Usage File
------------------------------
Demonstrates 3 common usage patterns:
1. Minimal default usage
2. Custom data with basic parameters
3. Full parameter customization
"""

import numpy as np
from phyto_nas_tsc import fit

def minimal_example():
    #Simplest possible usage with package-provided data
    print("\n=== Running Minimal Example ===")
    from importlib.resources import files
    data_path = str(files('phyto_nas_tsc.data'))  # Convert path here
    
    result = fit(data_dir='phyto_nas_tsc/data')
    print(f"\nBest model accuracy: {result['accuracy']:.4f}")

def custom_data_example():
    #Using custom synthetic data with basic parameters
    print("\n=== Running Custom Data Example ===")
    
    # Generate synthetic time series data
    np.random.seed(42)
    X = np.random.randn(150, 1, 50)  # 150 samples, 1 timestep, 50 features
    y = np.zeros((150, 3))           # 3 classes
    y[:50, 0] = 1   # First 50 samples = class 0
    y[50:100, 1] = 1 # Next 50 = class 1
    y[100:, 2] = 1   # Last 50 = class 2

    result = fit(
        X=X, 
        y=y,
        others={
            'population_size': 8,
            'generations': 2,
            'early_stopping': True
        }
    )
    
    print("\n=== Results ===")
    print(f"Best Accuracy: {result['accuracy']:.4f}")
    print(f"Model Config: {result['architecture']}")

def full_customization_example():
    """Demonstrates all customizable parameters"""
    print("\n=== Running Full Customization Example ===")
    
    result = fit(
        scoring='accuracy',
        data_dir='./custom_data',  # Alternative data location
        others={
            # Evolutionary parameters
            'population_size': 12,
            'generations': 4,
            
            # Resource management
            'timeout': 1800,  # 30 minute limit
            'max_iterations': 200,
            
            # Early stopping
            'early_stopping': True,
            'target_accuracy': 0.95,
            
            # Evolutionary operators
            'initial_F': 0.9,
            'final_F': 0.2,
            'initial_CR': 0.95,
            'final_CR': 0.3,
            
            # Model training
            'batch_size': 64,
            'learning_rate': 1e-4
        }
    )
    
    print("\n=== Full Results ===")
    print(f"Accuracy: {result['accuracy']:.4f}")
    print(f"Fitness: {result['fitness']:.4f}")
    print("Full architecture configuration:")
    for key, value in result['architecture'].items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    # Run all examples
    minimal_example()
    custom_data_example()
    full_customization_example()
    
    print("\nAll examples completed successfully!")