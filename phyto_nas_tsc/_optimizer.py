import numpy as np
from typing import Dict, Any
from phyto_nas_tsc._evolutionary_algorithm import NASDifferentialEvolution
from ._model_builder import build_model
from ._data_handler import DataHandler
from ._utils import fitness_function, save_results_csv

class NASOptimizer:
    def __init__(self, scoring='accuracy', population_size=15, generations=5, verbose=True, **others):
        self.scoring = scoring
        self.population_size = population_size
        self.generations = generations
        self.verbose = verbose
        self.others = others
        
    def optimize(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Core optimization routine"""
        # Initialize your evolutionary algorithm
        nas = NASDifferentialEvolution(
            population_size=self.population_size,
            generations=self.generations,
            verbose=self.verbose,
            **self.others
        )
        
        # Run optimization
        best_model = nas.evolve_and_check(X, y, input_size=X.shape[1])
        
        return {
            'architecture': best_model,
            'accuracy': nas.best_accuracy,
            'fitness': nas.best_fitness,
            'history': nas.history,
            'parameters': {'population_size': self.population_size, 'generations': self.generations, **self.others}
        }