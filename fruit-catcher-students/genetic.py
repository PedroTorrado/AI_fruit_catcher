import random
import numpy as np
from game import get_score

def create_individual(individual_size):
    """Create a single individual with random weights.
    
    Args:
        individual_size (int): Size of the individual (number of weights)
        
    Returns:
        list: A list of random weights between -1 and 1
    """
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    """Generate initial population of individuals.
    
    Args:
        individual_size (int): Size of each individual
        population_size (int): Number of individuals to generate
        
    Returns:
        list: List of randomly generated individuals
    """
    return [create_individual(individual_size) for _ in range(population_size)]

def select_parents(population, scores, elite_size):
    """Select parents using elitism and tournament selection.
    
    Args:
        population (list): List of individuals
        scores (list): Fitness scores for each individual
        elite_size (int): Number of top individuals to preserve
        
    Returns:
        list: Selected parent individuals
    """
    sorted_pairs = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
    sorted_population = [pair[0] for pair in sorted_pairs]
    sorted_scores = [pair[1] for pair in sorted_pairs]

    parents = sorted_population[:elite_size]

    while len(parents) < len(population):
        tournament_indices = random.sample(range(len(sorted_population)), 8)
        best_idx = max(tournament_indices, key=lambda i: sorted_scores[i])
        winner = sorted_population[best_idx]
        parents.append(winner)

    return parents

def crossover(parent1, parent2):
    """Perform one-point crossover between two parents.
    
    Args:
        parent1 (list): First parent's weights
        parent2 (list): Second parent's weights
        
    Returns:
        list: Child weights created from parents
    """
    point = random.randint(0, len(parent1))
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate, generation=None, max_generations=None):
    """Apply simple random mutation to an individual's weights.
    
    Args:
        individual (list): Individual to mutate
        mutation_rate (float): Base mutation rate
        generation (int, optional): Not used
        max_generations (int, optional): Not used
        
    Returns:
        list: Mutated individual
    """
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Random adjustment between -0.5 and 0.5
            adjustment = random.uniform(-0.5, 0.5)
            individual[i] += adjustment
            individual[i] = max(-1, min(1, individual[i]))  # Clamp to [-1, 1]
    return individual

def inject_new_individuals(population, fitness_scores, individual_size, population_size, stagnation_count):
    """Inject new individuals into the population when stagnation is detected.
    
    Args:
        population: Current population of individuals
        fitness_scores: List of fitness scores for each individual
        individual_size: Size of each individual
        population_size: Total size of the population
        stagnation_count: Number of consecutive stagnation periods
        
    Returns:
        Updated population with new individuals
    """
    # Progressive injection rates based on stagnation count
    injection_rate = min(0.3 + (stagnation_count * 0.1), 0.6)  # Start at 30%, max at 60%
    num_new_individuals = max(2, int(population_size * injection_rate))
    
    print(f"Stagnation detected! ({stagnation_count} consecutive periods). "
          f"Injecting {num_new_individuals} new individuals ({injection_rate*100:.0f}% of population).")
    
    # Keep best individuals and add new ones
    sorted_indices = np.argsort(fitness_scores)
    keep_size = len(population) - num_new_individuals
    population = [population[i] for i in sorted_indices[-keep_size:]]
    
    # Add new random individuals
    new_individuals = generate_population(individual_size, num_new_individuals)
    population.extend(new_individuals)
    
    return population

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness,
                      generations, elite_rate=0.05, mutation_rate=0.25, num_seeds=12):
    """Genetic algorithm with adaptive mutation, elitism, and tournament selection."""
    population = generate_population(individual_size, population_size)
    
    best_individual = None
    best_fitness = float('-inf')
    elite_size = max(1, int(population_size * elite_rate))
    fitness_history = []
    
    # Parameters for stagnation detection
    STAGNATION_WINDOW = 5  # Look at last 20 generations
    MIN_BEST_IMPROVEMENT = 0.01  # Require 1% improvement in best fitness
    MIN_AVG_IMPROVEMENT = 0.05  # Require 5% improvement in average fitness
    last_improvement_gen = 0
    best_fitness_history = []
    avg_fitness_history = []
    stagnation_count = 0
    
    for generation in range(generations):
        # Evaluate population
        fitness_scores = []
        for individual in population:
            try:
                # Use multiple seeds for evaluation via fitness function
                individual_scores = []
                for seed in range(num_seeds):
                    fitness = fitness_function(individual, seed=seed)
                    individual_scores.append(fitness)
                avg_fitness_score = sum(individual_scores) / num_seeds
                # Apply movement penalty based on score variance
                score_variance = np.var(individual_scores)
                movement_penalty = -2.0 if score_variance < 1.0 else 0.0
                adjusted_score = avg_fitness_score + movement_penalty
                fitness = max(0, adjusted_score)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
                    last_improvement_gen = generation
                    stagnation_count = 0  # Reset stagnation count on improvement
                    print(f"Generation {generation + 1}/{generations}, New Best Score: {fitness:.2f}")

                    if fitness >= target_fitness:
                        print(f"Target fitness {target_fitness} reached!")
                        return best_individual, best_fitness

            except Exception as e:
                print(f"Error evaluating individual: {e}")
                fitness_scores.append(float('-inf'))

        avg_fitness = np.mean([f for f in fitness_scores if f != float('-inf')])
        std_fitness = np.std([f for f in fitness_scores if f != float('-inf')])
        fitness_history.append((best_fitness, avg_fitness, std_fitness))
        best_fitness_history.append(best_fitness)
        avg_fitness_history.append(avg_fitness)
        
        print(f"Generation {generation + 1}: Best = {best_fitness:.2f}, Avg = {avg_fitness:.2f}, Std = {std_fitness:.2f}")

        # Check for stagnation less frequently and consider both best and average fitness
        if generation >= STAGNATION_WINDOW and generation % 5 == 0:  # Check every 5 generations
            recent_best = best_fitness_history[-STAGNATION_WINDOW:]
            recent_avg = avg_fitness_history[-STAGNATION_WINDOW:]
            
            # Calculate improvements
            best_improvement = (recent_best[-1] - recent_best[0]) / (recent_best[0] + 1e-6)
            avg_improvement = (recent_avg[-1] - recent_avg[0]) / (recent_avg[0] + 1e-6)
            
            # Consider stagnation if both best and average fitness show little improvement
            # Using different thresholds for best and average fitness
            if (best_improvement < MIN_BEST_IMPROVEMENT and 
                avg_improvement < MIN_AVG_IMPROVEMENT and 
                (generation - last_improvement_gen) >= STAGNATION_WINDOW):
                
                stagnation_count += 1
                print(f"Stagnation detected! Best improvement: {best_improvement:.3f} (threshold: {MIN_BEST_IMPROVEMENT:.3f}), "
                      f"Average improvement: {avg_improvement:.3f} (threshold: {MIN_AVG_IMPROVEMENT:.3f})")
                population = inject_new_individuals(population, fitness_scores, individual_size, 
                                                 population_size, stagnation_count)
                continue

        # Selection and next generation
        parents = select_parents(population, fitness_scores, elite_size)
        new_population = parents[:elite_size]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            # Increase mutation rate when stagnation is detected
            current_mutation_rate = mutation_rate * (1 + 0.5 * stagnation_count)
            child = mutate(child, current_mutation_rate, generation, generations)
            new_population.append(child)

        population = new_population
    
    # Summary of evolution progress
    print("\nEvolution Summary:")
    print(f"Initial Best Fitness (Gen 1): {fitness_history[0][0]:.2f}")
    print(f"Final Best Fitness (Gen {generations}): {best_fitness:.2f}")
    print(f"Initial Avg Fitness (Gen 1): {fitness_history[0][1]:.2f}")
    print(f"Final Avg Fitness (Gen {generations}): {avg_fitness:.2f}")
    
    return best_individual, best_fitness