import numpy as np
from phyto_nas_tsc import fit

# ---- Test for fit function ---- #
def test_fit_function():
    X = np.random.rand(50, 5, 1)
    y = np.eye(2)[np.random.randint(0, 2, 50)]
    
    result = fit(X, y, others={"generations": 3, "population_size": 4})
    
    assert isinstance(result, dict)
    assert 'architecture' in result
    assert 'accuracy' in result
    assert 0 <= result['accuracy'] <= 1

if __name__ == "__main__":
    test_fit_function()
    print("All tests passed!")