import numpy as np
from phyto_nas_tsc import fit

def test_fit_function():
    """Test the main fit function with dummy data"""
    X = np.random.rand(50, 5, 1)  # Smaller dataset for testing
    y = np.eye(2)[np.random.randint(0, 2, 50)]
    
    result = fit(X, y, generations=1, population_size=2)
    
    assert isinstance(result, dict)
    assert 'best_accuracy' in result
    assert 0 <= result['best_accuracy'] <= 1