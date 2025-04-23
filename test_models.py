import torch
from phyto_nas_tsc._model_builder import FCNN, CNN, GRU, LSTM, TransformerModel

# Test Fully Connected Neural Network (FCNN)
def test_fcnn():
    model = FCNN(input_size=10, hidden_units=64, output_size=2, num_layers=3)
    dummy_input = torch.randn(32, 10)  # (batch_size, input_size)
    output = model(dummy_input)
    assert output.shape == (32, 2), f"Unexpected output shape: {output.shape}"
    print("FCNN test passed!")

# Test CNN Model
def test_cnn():
    model = CNN(input_channels=1, num_filters=32, kernel_size=3, output_size=2)
    dummy_input = torch.randn(32, 1, 100)  # (batch_size, channels, time_steps)
    output = model(dummy_input)
    assert output.shape == (32, 2), f"Unexpected output shape: {output.shape}"
    print("CNN test passed.")

# Test GRU Model
def test_gru():
    model = GRU(input_size=10, hidden_units=128, output_size=2, num_layers=2)
    dummy_input = torch.randn(32, 50, 10)  # (batch_size, seq_len, input_size)
    output = model(dummy_input)
    assert output.shape == (32, 2), f"Unexpected output shape: {output.shape}"
    print("GRU test passed.")

# Test LSTM Model
def test_lstm():
    model = LSTM(input_size=10, hidden_units=128, output_size=2, num_layers=2)
    dummy_input = torch.randn(32, 50, 10)  # (batch_size, seq_len, input_size)
    output = model(dummy_input)
    assert output.shape == (32, 2), f"Unexpected output shape: {output.shape}"
    print("LSTM test passed.")

# Test Transformer Model
def test_transformer():
    model = TransformerModel(input_dim=10, num_heads=2, num_layers=2, hidden_dim=16, output_size=2)
    dummy_input = torch.randn(32, 50, 10)  # (batch_size, seq_len, input_dim)
    output = model(dummy_input)
    assert output.shape == (32, 2), f"Unexpected output shape: {output.shape}"
    print("Transformer test passed.")

if __name__ == "__main__":
    test_fcnn()
    test_cnn()
    test_gru()
    test_lstm()
    test_transformer()