from importlib.resources import files
from ._optimizer import NASOptimizer
from ._data_handler import DataHandler, validate_inputs

def fit(X=None, y=None, scoring='accuracy', data_dir=None, others=None):
    """
    Main API function for neural architecture search.
    
    Args:
        X: numpy.ndarray (n_samples, timesteps, features) or None
        y: numpy.ndarray One-hot encoded labels or None
        scoring: Metric to optimize ('accuracy')
        data_dir: Path to data if X/y not provided
        **kwargs: Additional optimization parameters
    """

    others = others or {}
    population_size = others.get("population_size", 5)
    generations = others.get("generations", 2)

    if population_size < 3:
        raise ValueError("population_size must be at least 3 for evolution")
    
    # Load data if X/y not provided
    if X is None or y is None:
        if data_dir is None:
            data_dir = files('phyto_nas_tsc.data')  # Use package data
        handler = DataHandler(data_dir)
        handler.load_and_preprocess()
        X = handler.X_analysis
        y = handler.y_analysis
    
    # Now validate the loaded data
    validate_inputs(X, y)
    
    if len(X) < 5:  # Check sample size after data is loaded
        raise ValueError("Need at least 5 samples for evolution")
    
    optimizer = NASOptimizer(
        scoring=scoring,
        population_size=population_size,
        generations=generations,
        **others
    )
    return optimizer.optimize(X, y)