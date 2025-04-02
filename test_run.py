import os
import numpy as np
from phyto_nas_tsc import fit
from phyto_nas_tsc._data_handler import DataHandler

def validate_and_reshape_data(handler):
    """Ensure data is properly shaped before training"""
    X = handler.X_analysis
    y = handler.y_analysis
    
    print("\n=== Data Validation ===")
    print(f"Original X shape: {X.shape}")
    print(f"Original y shape: {y.shape}")
    
    # Calculate expected features
    total_elements = np.prod(X.shape)
    n_samples = X.shape[0]
    expected_features = total_elements // n_samples
    
    # Reshape X to 3D (samples, timesteps, features)
    if len(X.shape) == 2:
        try:
            X = X.reshape(n_samples, 1, expected_features)
            print(f"Reshaped X to: {X.shape}")
        except ValueError as e:
            print(f"Reshape failed. Total elements: {total_elements}")
            print(f"Can't reshape into ({n_samples}, 1, {expected_features})")
            raise
    
    # Validate y is one-hot encoded
    if len(y.shape) != 2:
        raise ValueError(f"y must be 2D (one-hot), got shape {y.shape}")
    
    return X, y

def main():
    data_dir = os.path.join(os.path.dirname(__file__), "phyto_nas_tsc", "data")
    
    try:
        print("\n=== Loading Data ===")
        handler = DataHandler(data_dir)
        handler.load_and_preprocess()
        
        X, y = validate_and_reshape_data(handler)
        
        print("\n=== Starting Optimization ===")
        result = fit(
            X=X,
            y=y,
            population_size=8,
            generations=3,
            scoring='accuracy'
        )
        
        print("\n=== Results ===")
        print(f"Best Accuracy: {result['accuracy']:.4f}")
        print(f"Model Config: {result['architecture']}")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        print("Troubleshooting:")
        print("- Check X_train.csv has consistent dimensions")
        print("- Verify y_train.csv is one-hot encoded")
        print("- Ensure no missing values in data")

if __name__ == "__main__":
    main()