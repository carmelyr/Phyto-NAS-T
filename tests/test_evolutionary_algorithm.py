import pytest
from phyto_nas_tsc._evolutionary_algorithm import NASDifferentialEvolution
from phyto_nas_tsc._config import population_size, generations

# This file contains tests for the evolutionary algorithm in the phyto_nas_tsc package

# ---- Test for NASDifferentialEvolution Class ---- #
def test_nas_de_initialization():
    nas = NASDifferentialEvolution(verbose=False)
    assert len(nas.population) == population_size
    assert nas.generations == generations

def test_initialize_population():
    nas = NASDifferentialEvolution(verbose=False)
    population = nas.initialize_population()
    assert len(population) == population_size
    for individual in population:
        assert "hidden_units" in individual
        assert individual["hidden_units"] in [64, 128, 256, 512]
        assert "num_layers" in individual
        assert 1 <= individual["num_layers"] <= 8

def test_mutate(small_population):
    nas = NASDifferentialEvolution(verbose=False)
    parent1, parent2, parent3 = small_population[0], small_population[1], small_population[0]
    mutant = nas.mutate(parent1, parent2, parent3)
    
    assert "hidden_units" in mutant
    assert 64 <= mutant["hidden_units"] <= 512
    assert "num_layers" in mutant
    assert 1 <= mutant["num_layers"] <= 8

def test_crossover(small_population):
    nas = NASDifferentialEvolution(verbose=False)
    parent = small_population[0]
    mutant = small_population[1]
    offspring = nas.crossover(parent, mutant)
    
    # checks that offspring has some properties from both parents
    different = False
    for key in parent:
        if key in mutant and parent[key] != mutant[key]:
            if offspring[key] == mutant[key]:
                different = True
                break
    assert different, "Offspring should have some properties from mutant"