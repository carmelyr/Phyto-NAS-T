import numpy as np
from typing import Dict, Any
from phyto_nas_tsc._evolutionary_algorithm import NASDifferentialEvolution
from ._config import population_size, generations

# ---- Optimizer Class ---- #
"""
- it is a wrapper for the NASDifferentialEvolution class
- it initializes the class with the given parameters
- it provides a method to optimize the architecture
"""
class NASOptimizer:
    def __init__(self, scoring='accuracy', verbose=True, **others):
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        self.others = others
        
    # This method is used to set the parameters for the optimizer
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        # filters out population_size and generations from self.others to avoid duplication
        filtered_others = {k: v for k, v in self.others.items() if k not in ['population_size', 'generations']}
        
        nas = NASDifferentialEvolution(
            population_size=self.population_size,
            generations=self.generations,
            verbose=self.verbose,
            **filtered_others
        )
        
        # runs optimization
        best_model = nas.evolve_and_check(X, y, input_size=X.shape[1])
        
        return {
            'architecture': best_model,
            'accuracy': nas.best_accuracy,
            'fitness': nas.best_fitness,
            'history': nas.history,
            'parameters': {'population_size': self.population_size, 'generations': self.generations, **self.others}
        }