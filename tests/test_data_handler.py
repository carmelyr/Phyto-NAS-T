import pytest
import numpy as np
from phyto_nas_tsc._data_handler import DataHandler, validate_inputs

# ---- Test for DataHandler and validate_inputs ---- #
def test_validate_inputs_valid(dummy_data):
    X, y = dummy_data
    validate_inputs(X, y)

# Test for invalid input shapes
def test_validate_inputs_invalid_shapes():
    X = np.random.rand(50, 5)
    y = np.random.rand(50, 2)
    with pytest.raises(ValueError):
        validate_inputs(X, y)
    
    X = np.random.rand(50, 1, 5)
    y = np.random.rand(49, 2)
    with pytest.raises(ValueError):
        validate_inputs(X, y)

def test_data_handler_initialization():
    handler = DataHandler()
    assert handler.data_dir.name == 'classification_ozone'
    
    custom_handler = DataHandler(data_dir='custom_data')
    assert custom_handler.data_dir.name == 'custom_data'

def test_load_and_preprocess(tmp_path):
    import pandas as pd
    import os
    
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    
    # creates dummy data
    X_train = pd.DataFrame(np.random.rand(10, 5))
    y_train = pd.DataFrame(np.random.randint(0, 2, 10))
    X_test = pd.DataFrame(np.random.rand(5, 5))
    y_test = pd.DataFrame(np.random.randint(0, 2, 5))
    
    # saves to CSV
    X_train.to_csv(data_dir / "X_train.csv", index=False)
    y_train.to_csv(data_dir / "y_train.csv", index=False)
    X_test.to_csv(data_dir / "X_test.csv", index=False)
    y_test.to_csv(data_dir / "y_test.csv", index=False)
    
    handler = DataHandler(data_dir=str(data_dir))
    handler.load_and_preprocess()
    
    assert handler.X_analysis.shape == (10, 1, 5)
    assert handler.y_analysis.shape == (10, 2)
    assert handler.X_test.shape == (5, 1, 5)
    assert handler.y_test.shape == (5, 2)