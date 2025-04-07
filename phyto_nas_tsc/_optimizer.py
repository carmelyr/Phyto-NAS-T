import numpy as np
from typing import Dict, Any
from phyto_nas_tsc._evolutionary_algorithm import NASDifferentialEvolution

# ---- Optimizer Class ---- #
"""
- it is a wrapper for the NASDifferentialEvolution class
- it initializes the class with the given parameters
- it provides a method to optimize the architecture
"""
class NASOptimizer:
    def __init__(self, scoring='accuracy', verbose=True, **others):
        self.scoring = scoring
        self.verbose = verbose
        self.others = others
        
    # This method is used to set the parameters for the optimizer
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:

        nas = NASDifferentialEvolution(
            verbose=self.verbose,
            **self.others
        )
        
        # runs optimization
        best_model = nas.evolve_and_check(X, y, input_size=X.shape[1])
        
        return {
            'architecture': best_model,
            'accuracy': nas.best_accuracy,
            'fitness': nas.best_fitness,
            'history': nas.history,
            'parameters': self.others
        }