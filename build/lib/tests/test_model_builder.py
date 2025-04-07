import pytest
import torch
from phyto_nas_tsc._model_builder import build_model, LSTM

# This file contains tests for the model builder in the phyto_nas_tsc package
def test_build_model(dummy_data):
    X, y = dummy_data
    model = build_model(
        model_type="LSTM",
        input_size=X.shape[-1],
        hidden_units=64,
        output_size=y.shape[1],
        num_layers=2,
        dropout_rate=0.2,
        bidirectional=True,
        attention=True,
        learning_rate=1e-3,
        weight_decay=1e-5
    )
    assert isinstance(model, LSTM)

# Test the LSTM model with different configurations
def test_lstm_forward_pass(dummy_data):
    X, y = dummy_data
    model = LSTM(
        input_size=X.shape[-1],
        hidden_units=64,
        output_size=y.shape[1],
        num_layers=2
    )
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    output = model(X_tensor)
    
    assert output.shape == (X.shape[0], y.shape[1])

def test_lstm_training_step(dummy_data):
    X, y = dummy_data
    model = LSTM(
        input_size=X.shape[-1],
        hidden_units=64,
        output_size=y.shape[1]
    )
    
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)
    
    loss = model.training_step((X_tensor, y_tensor), 0)
    assert isinstance(loss, torch.Tensor)