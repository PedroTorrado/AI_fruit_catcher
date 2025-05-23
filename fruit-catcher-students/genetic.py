import random
import numpy as np
from game import get_score

def create_individual(individual_size):
    """Create a single individual with random weights."""
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    """Generate initial population."""
    return [create_individual(individual_size) for _ in range(population_size)]

def evaluate_population(population, network):
    """Evaluate each individual."""
    scores = []
    for individual in population:
        network.load_weights(individual)
        score = get_score(network.forward)
        scores.append(score)
    return scores

def select_parents(population, scores, elite_size):
    """Select parents using elitism and tournament selection."""
    # Sort population by fitness
    sorted_pairs = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    sorted_population = [pair[0] for pair in sorted_pairs]
    
    # Keep elite individuals
    parents = sorted_population[:elite_size]
    
    # Select rest through tournament selection
    while len(parents) < len(population):
        # Tournament selection with size 3
        tournament = random.sample(sorted_population, 3)
        tournament_scores = [scores[population.index(ind)] for ind in tournament]
        winner = tournament[tournament_scores.index(max(tournament_scores))]
        parents.append(winner)
    
    return parents

def crossover(parent1, parent2):
    """Simple one-point crossover."""
    point = random.randint(0, len(parent1))
    child = parent1[:point] + parent2[point:]
    return child

def mutate(individual, mutation_rate):
    """Simple mutation."""
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] += random.gauss(0, 0.2)  # Small random change
    return individual

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness, generations, elite_rate=0.2, mutation_rate=0.05):
    """Simple genetic algorithm."""
    # Initialize population
    population = generate_population(individual_size, population_size)
    best_individual = None
    best_fitness = float('-inf')
    
    # Calculate elite size
    elite_size = max(1, int(population_size * elite_rate))
    
    for generation in range(generations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            try:
                fitness = fitness_function(individual)
                fitness_scores.append(fitness)
                
                # Update best individual
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    print(f"Generation {generation + 1}/{generations}, New Best Score: {fitness:.2f}")
                    
                    # Check if target reached
                    if fitness >= target_fitness:
                        print(f"Target fitness {target_fitness} reached!")
                        return best_individual, best_fitness
                        
            except Exception as e:
                print(f"Error evaluating individual: {e}")
                fitness_scores.append(float('-inf'))
        
        # Select parents
        parents = select_parents(population, fitness_scores, elite_size)
        
        # Create new population starting with elite individuals
        new_population = parents[:elite_size]
        
        # Fill rest of population with crossover and mutation
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate)
            new_population.append(child)
        
        population = new_population
    
    return best_individual, best_fitness