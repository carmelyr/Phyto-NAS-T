from importlib.resources import files
from ._optimizer import NASOptimizer
from ._data_handler import DataHandler, validate_inputs

# ---- Main API Function ---- #
def fit(X=None, y=None, scoring='accuracy', data_dir=None, others=None):
    """
    Args:
        X: numpy.ndarray (n_samples, timesteps, features) or None
        y: numpy.ndarray One-hot encoded labels or None
        scoring: Metric to optimize ('accuracy')
        data_dir: Path to data if X/y not provided
        others: Dict of additional optimization parameters including:
            - population_size: number of individuals in population
            - generations: number of evolutionary generations
            - timeout: maximum time limit (in seconds)
            - early_stopping: whether to use early stopping
            - max_iterations: maximum training iterations
    """
    others = others or {}
    population_size = others.get("population_size", 10)
    generations = others.get("generations", 5)
    
    # validates parameters
    if population_size < 3:
        raise ValueError("population_size must be at least 3 for evolution")
    
    # loads data from file if X and y are not provided and data_dir is None
    # data_dir=None triggers package data
    if X is None or y is None:
        handler = DataHandler(data_dir=data_dir)
        handler.load_and_preprocess()
        X = handler.X_analysis
        y = handler.y_analysis
    
    # validates the loaded data
    validate_inputs(X, y)
    
    if len(X) < 5:
        raise ValueError("Need at least 5 samples for evolution")
    
    optimizer = NASOptimizer(
        scoring=scoring,
        **others
    )
    result = optimizer.optimize(X, y)
   
    if 'fitness' in result['architecture']:
        del result['architecture']['fitness']
    if 'accuracy' in result['architecture']:
        del result['architecture']['accuracy']
    
    return result