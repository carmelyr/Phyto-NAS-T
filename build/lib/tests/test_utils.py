import pytest
import os
import tempfile
from phyto_nas_tsc._utils import save_results_csv, fitness_function

def test_save_results_csv():
    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        filename = tmp.name
    
    try:
        save_results_csv(
            filename, 1, 1, "LSTM", "{'layers':2}", 
            [0.8, 0.9], 0.85, 10000, 120.5
        )
        
        save_results_csv(
            filename, 2, 1, "LSTM", "{'layers':3}", 
            [0.7, 0.8], 0.75, 15000, 180.2
        )
        
        with open(filename, 'r') as f:
            lines = f.readlines()
            assert len(lines) == 3
            assert "Run ID" in lines[0]
            assert "1,1,LSTM" in lines[1]
            assert "2,1,LSTM" in lines[2]
    finally:
        os.unlink(filename)

def test_fitness_function():
    fitness1 = fitness_function("LSTM", 0.9, 1000000, 60)
    
    fitness2 = fitness_function("LSTM", 0.7, 1000000, 60)
    
    fitness3 = fitness_function("LSTM", 0.9, 5000000, 60)
    
    fitness4 = fitness_function("LSTM", 0.9, 1000000, 300)
    
    assert fitness1 > fitness2, "Higher accuracy should give higher fitness"
    assert fitness1 > fitness3, "Smaller model should give higher fitness"
    assert fitness1 > fitness4, "Faster training should give higher fitness"
    assert fitness1 >= 0, "Fitness should never be negative"