# Unit Tests

This folder contains unit tests for the Phyto Neural Architecture Search for Time Series Classification framework.
They are designed to ensure the correctness and robustness of the core components.

## Test Files Overview

### conftest.py
Includes shared setup used in other tests, like:
- Sample LSTM models
- Dummy data and conigurations

### test_core.py
Tests the core functionality of the evolutionary algorithm:
- Evolutionary loop
- Population generation and selection
- Evaluation logic

### test_data_handler.py
Validates preprocessing and data handling utilities:
- Normalization
- Sequence formatting
- Train/test split integrity

### test_evolutionary_algorithm.py
Focuses on evolutionary operations:
- Mutation and crossover
- Fitness evaluation
- Logging and traceback recording on failure

### test_model_builder.py
Tests dynamic model construction:
- LSTM-based model creation
- Layer configuration and architecture validation

### test_utils.py
Covers helper utilities:
- Scoring functions (e.g. accuracy)
- Time measurement
- Model checkpoint handling

## Running the Tests

All tests can be run using `pytest` from the root directory:

```bash
pytest
