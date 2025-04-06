import torch

# Hyperparameters
population_size = 10    # number of individuals in the population
generations = 5         # number of generations
initial_F = 0.8         # initial mutation factor
final_F = 0.3           # final mutation factor
initial_CR = 0.9        # initial crossover rate
final_CR = 0.4          # final crossover rate
decay_rate = 0.85       # exponential decay rate
alpha = 0.0000001       # size penalty
BETA = 0.00000001       # time penalty
num_folds = 5           # number of folds for cross-validation
max_iterations = 1000   # maximum number of training iterations
early_stopping = True  # whether to use early stopping
timeout = 3600         # maximum time limit in seconds
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")