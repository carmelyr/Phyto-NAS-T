import pytest
import numpy as np

# This file contains fixtures and test data for the phyto_nas_tsc package
# Fixtures are used to set up the environment for testing
@pytest.fixture
def dummy_data():
    X = np.random.rand(50, 1, 5)                            # 50 samples, 1 time step, 5 features
    y = np.eye(2)[np.random.randint(0, 2, 50)]              # One-hot encoded
    return X, y

@pytest.fixture
def small_population():
    return [
        {
            "model_type": "LSTM",
            "hidden_units": 64,
            "num_layers": 2,
            "dropout_rate": 0.2,
            "bidirectional": True,
            "learning_rate": 1e-3,
            "batch_size": 32,
            "weight_decay": 1e-5,
            "attention": False
        },
        {
            "model_type": "LSTM",
            "hidden_units": 128,
            "num_layers": 3,
            "dropout_rate": 0.3,
            "bidirectional": False,
            "learning_rate": 1e-4,
            "batch_size": 64,
            "weight_decay": 0,
            "attention": True
        }
    ]