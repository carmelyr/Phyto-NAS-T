import os
from ._config import alpha, BETA
import csv

# This function saves the results of the model training to a CSV file.
# it appends the results to the file if it already exists, or creates a new file if it doesn't.
def save_results_csv(filename, run_id, generation, architecture, layers, fold_accuracies, val_accuracy, model_size, runtime):
    file_exists = os.path.exists(filename)
    
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        
        # Add headers if file is newly created or empty
        if not file_exists or os.stat(filename).st_size == 0:
            writer.writerow(["Run ID", "Generation", "Architecture", "Layers", "Fold Accuracies", "Validation Accuracy", "Model Size", "Training Time (s)"])
        
        # Write the new row with the provided run_id
        writer.writerow([run_id, generation, architecture, layers, fold_accuracies, val_accuracy, model_size, runtime])

# This function calculates the fitness of a model based on its architecture, validation accuracy, model size, and training time.
"""
- architecture: The architecture of the model (e.g., number of layers, types of layers).
- validation_accuracy: The accuracy of the model on the validation set.
- model_size: The size of the model in terms of number of parameters.
- training_time: The time taken to train the model.
- alpha: A hyperparameter that penalizes larger models.
- BETA: A hyperparameter that penalizes longer training times.
- fitness: The calculated fitness score of the model.
"""
def fitness_function(architecture, validation_accuracy, model_size, training_time):
    accuracy_weight = 1.0
    size_penalty = alpha * (model_size / 1e6)   # scales the model size to millions of parameters
    time_penalty = BETA * (training_time / 60)  # scales the training time to minutes
    
    # squares the validation accuracy to emphasize its importance in the fitness score
    accuracy_score = validation_accuracy ** 2
    
    fitness = (accuracy_weight * accuracy_score) - size_penalty - time_penalty
    return max(0.0, fitness)