import torch
import numpy as np
import pytorch_lightning as pl
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
from ._model_builder import build_model
from ._utils import fitness_function, save_results_csv
from ._config import initial_F, final_F, initial_CR, final_CR, decay_rate
from pytorch_lightning.callbacks import EarlyStopping
import random
import time
import traceback
from tqdm import tqdm
import sys
import csv
import os

# ---- Evolutionary Algorithm Class ---- #
"""
- This class implements a Differential Evolution algorithm for Neural Architecture Search (NAS).
- initializes a population of models with random hyperparameters
- evolves the population over a number of generations
- uses mutation and crossover to create new models
- evaluates the models using cross-validation
- selects the best models based on a fitness function
- saves the results to a CSV file
"""
class NASDifferentialEvolution:
    def __init__(self, verbose=True, **others):
        self.population_size = others.get('population_size', 10)
        self.generations = others.get('generations', 5)
        self.population = self.initialize_population()
        self.initial_F = initial_F
        self.final_F = final_F
        self.initial_CR = initial_CR
        self.final_CR = final_CR
        self.decay_rate = decay_rate

        self.best_fitness = -float('inf')
        self.best_model = None
        self.best_accuracy = 0.0

        self.verbose = verbose
        self.others = others
        self.timeout = others.get('timeout', None)
        self.early_stopping = others.get('early_stopping', False)
        self.max_iterations = others.get('max_iterations', 100)

        self.history = []
        self.total_generations = self.generations
        self.run_id = self.get_next_run_id("evolution_results.csv")

    # This method is used to calculate the current mutation factor and crossover rate
    def get_current_rates(self, generation):

        # hybrid mutation factor calculation
        if generation < self.total_generations // 2:
            
            # uses exponential decay for the first half of the generations
            current_F = max(self.final_F, self.initial_F * (decay_rate ** generation))

        else:
            # uses linear decay for the second half of the generations
            # calculates the linear progress from the midpoint to the end and interpolates the mutation factor
            linear_progress = (generation - self.total_generations//2) / (self.total_generations//2)
            current_F = initial_F - (initial_F - final_F) * linear_progress

        # crossover rate calculation
        current_CR = initial_CR - (initial_CR - final_CR) * (generation / self.total_generations)
        
        return current_F, current_CR

    # This method is used to get the next run_id for the results file
    def get_next_run_id(self, results_file):
        """
        - reads the last run_id from the results file and increments it.
        - if the file does not exist or is empty, it starts with run_id = 1.
        """
        if not os.path.exists(results_file):
            return 1
        try:
            with open(results_file, "r") as f:
                reader = csv.reader(f)
                rows = list(reader)
                if len(rows) <= 1:
                    return 1
                last_run_id = int(rows[-1][0])
                return last_run_id + 1
            
        except Exception as e:
            print(f"Error reading run_id from {results_file}: {e}")
            return 1

    # This method is used to build the model based on the given configuration
    def build_model(self, model_config, X_train, y_train):
        return build_model(
            model_type=model_config["model_type"],
            input_size=X_train.shape[-1],
            hidden_units=int(model_config["hidden_units"]),
            output_size=y_train.shape[1],
            num_layers=int(model_config["num_layers"]),
            dropout_rate=model_config["dropout_rate"],
            bidirectional=model_config["bidirectional"],
            attention=model_config["attention"],
            learning_rate=model_config["learning_rate"],
            weight_decay=model_config["weight_decay"]
        )

    # This method is used to initialize the population with random hyperparameters
    # Each individual in the population is a dictionary with hyperparameters
    def initialize_population(self):
        population = []
        for _ in range(self.population_size):
            population.append({
                "model_type": "LSTM",
                "hidden_units": random.choice([64, 128, 256, 512]),
                "num_layers": random.choice([1, 2, 3, 4, 5, 6, 7, 8]),       
                "dropout_rate": random.uniform(0.1, 0.5),      
                "bidirectional": random.choice([True, False]),
                "learning_rate": random.choice([1e-5, 1e-4, 1e-3, 3e-4, 5e-5]), 
                "batch_size": random.choice([32, 64, 128]),
                "weight_decay": random.choice([0, 1e-5, 5e-5]),
                "attention": random.choice([True, False]),
            })
        return population

    # This method is used to mutate the parents to create a mutant
    # it uses the Differential Evolution strategy to create a new individual
    def mutate(self, parent1, parent2, parent3, current_F):
        mutant = {
            "model_type": "LSTM",
            "hidden_units": max(64, min(512, int(parent1["hidden_units"] + current_F * (parent2["hidden_units"] - parent3["hidden_units"])))),
            "num_layers": max(1, min(4, int(round(parent1["num_layers"] + current_F * (parent2["num_layers"] - parent3["num_layers"]))))),
            "dropout_rate": max(0.1, min(0.5, parent1["dropout_rate"] + current_F * (parent2["dropout_rate"] - parent3["dropout_rate"]))),
            "bidirectional": random.choice([parent1["bidirectional"], parent2["bidirectional"], parent3["bidirectional"]]),
            "attention": random.choice([parent1["attention"], parent2["attention"], parent3["attention"]]),
            "learning_rate": 10**(np.log10(parent1["learning_rate"]) + current_F * (np.log10(parent2["learning_rate"]) - np.log10(parent3["learning_rate"]))),
            "batch_size": random.choice([parent1["batch_size"], parent2["batch_size"], parent3["batch_size"]]),
            "weight_decay": random.choice([parent1["weight_decay"], parent2["weight_decay"], parent3["weight_decay"]])
        }
        return mutant

    # This method is used to perform crossover between the parent and mutant
    # it creates an offspring by combining the parent and mutant based on the crossover rate
    def crossover(self, parent, mutant, current_CR):
        offspring = parent.copy()
        for key in mutant:
            if random.random() < current_CR:
                offspring[key] = mutant[key]
        return offspring
    
    # This method is used to evaluate the model using cross-validation
    # it calculates the accuracy of the model on the validation set
    def evaluate_model(self, model, val_loader):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, y in val_loader:
                outputs = model(X)
                predicted = torch.argmax(outputs, dim=1)
                true_labels = torch.argmax(y, dim=1)
                correct += (predicted == true_labels).sum().item()
                total += y.size(0)
        accuracy = correct / total
        return accuracy

    # This method is used to perform cross-validation on the model
    # it splits the data into training and validation sets
    def cross_validate(self, model_config, X, y, input_size, generation):

        kf = KFold(n_splits=5, shuffle=True, random_state=42)

        scores = []
        fold_accuracies = []
        model_sizes = []
        training_times = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"\n--- Fold {fold + 1} ---")
            try:
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if X_train.ndim == 2:
                    features = X_train.shape[1]
                    X_train = X_train.reshape(-1, 1, features)
                    X_val = X_val.reshape(-1, 1, features)
                elif X_train.ndim != 3:
                    raise ValueError(
                        f"Input must be 2D or 3D, got {X_train.ndim}D array"
                    )

                if X_train.ndim == 2:
                    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])

                if X_val.ndim == 2:
                    X_val = X_val.reshape(X_val.shape[0], 1, X_val.shape[1])

                X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
                X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                y_val_tensor = torch.tensor(y_val, dtype=torch.float32)

                train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                train_loader = DataLoader(train_dataset, batch_size=model_config["batch_size"], shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=model_config["batch_size"], shuffle=False)

                model = self.build_model(model_config, X_train, y_train)

                trainer = pl.Trainer(
                    max_epochs=150,
                    enable_checkpointing=False,
                    callbacks=[
                        EarlyStopping(
                            monitor="val_acc",
                            patience=30,
                            mode="max",
                            min_delta=0.001,
                            stopping_threshold=0.9
                        )
                    ],
                    enable_progress_bar=False,
                    logger=False
                )

                start_time = time.time()
                trainer.fit(model, train_loader, val_loader)
                training_time = time.time() - start_time

                val_acc = self.evaluate_model(model, val_loader)
                model_size = sum(p.numel() for p in model.parameters() if p.requires_grad)
                fitness = fitness_function("LSTM", val_acc, model_size, training_time)

                print(f"\nFold {fold+1} Accuracy: {val_acc:.4f}, Fitness: {fitness:.4f}, Size: {model_size}, Time: {training_time:.2f}s")

                scores.append(fitness)
                fold_accuracies.append(val_acc)
                model_sizes.append(model_size)
                training_times.append(training_time)

            except Exception as e:
                print(f"Error in fold {fold + 1}: {str(e)}")
                traceback.print_exc()
                return -float('inf'), 0.0, 0, 0.0

        avg_fitness = np.mean(scores)
        avg_accuracy = np.mean(fold_accuracies)
        avg_model_size = np.mean(model_sizes)
        avg_training_time = np.mean(training_times)

        print(f"\n>>> Generation {generation+1}: Avg Accuracy = {avg_accuracy:.4f}, Avg Fitness = {avg_fitness:.4f}\n")

        if avg_accuracy > self.best_accuracy:
            self.best_accuracy = avg_accuracy
            self.best_model = model_config
            self.best_fitness = avg_fitness

        clean_config = model_config.copy()
        if 'fitness' in clean_config:
            del clean_config['fitness']
        if 'accuracy' in clean_config:
            del clean_config['accuracy']

        # saves results to CSV
        save_results_csv(
            "evolution_results.csv",
            self.run_id,
            generation + 1,
            "LSTM",
            str(clean_config),
            fold_accuracies,
            avg_accuracy,
            avg_model_size,
            avg_training_time
        )

        return avg_fitness, avg_accuracy, avg_model_size, avg_training_time

    # This method is used to evolve the population over generations
    # it selects parents, mutates them, performs crossover, and evaluates the offspring
    # it updates the population with the best individuals
    def evolve_and_check(self, X, y, input_size):
        for generation in tqdm(range(self.generations), desc="Evolution Progress", file=sys.stdout, dynamic_ncols=True):
            new_population = []
            current_F, current_CR = self.get_current_rates(generation)

            for i in range(self.population_size):
                candidates = [idx for idx in range(self.population_size) if idx != i]

                if len(candidates) < 3:
                    new_population.append(self.population[i])
                    continue

                # Selects 3 random parents for mutation
                parent1_idx, parent2_idx, parent3_idx = random.sample(candidates, 3)
                parent1 = self.population[parent1_idx]
                parent2 = self.population[parent2_idx]
                parent3 = self.population[parent3_idx]

                # Pass current_F to the mutate method
                mutant = self.mutate(parent1, parent2, parent3, current_F)
                offspring = self.crossover(self.population[i], mutant, current_CR)

                fitness, accuracy, model_size, training_time = self.cross_validate(offspring, X, y, input_size, generation)

                if fitness > self.population[i].get('fitness', -float('inf')):
                    offspring['fitness'] = fitness
                    offspring['accuracy'] = accuracy
                    new_population.append(offspring)
                else:
                    new_population.append(self.population[i])

            self.population = sorted(new_population, key=lambda x: x['fitness'], reverse=True)

            self.history.append({
                'generation': generation + 1,
                'best_fitness': self.population[0]['fitness'],
                'best_accuracy': self.population[0]['accuracy'],
                'best_model': self.population[0]
            })

            if self.verbose:
                print(f"\nGeneration {generation + 1} Best:")
                print(f"Fitness: {self.population[0]['fitness']:.4f}")
                print(f"Accuracy: {self.population[0]['accuracy']:.4f}")
                best_model = self.population[0].copy()
                if 'fitness' in best_model:
                    del best_model['fitness']
                if 'accuracy' in best_model:
                    del best_model['accuracy']
                print(f"Model: {best_model}\n")

        return self.population[0]