import random
import numpy as np
from game import get_score

def create_individual(individual_size):
    """Create a single individual with random weights."""
    return [random.uniform(-1, 1) for _ in range(individual_size)]

def generate_population(individual_size, population_size):
    """Generate initial population."""
    return [create_individual(individual_size) for _ in range(population_size)]

def select_parents(population, scores, elite_size):
    """Select parents using elitism and tournament selection."""
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
    """Simple one-point crossover."""
    point = random.randint(0, len(parent1))
    return parent1[:point] + parent2[point:]

def mutate(individual, mutation_rate, generation=None, max_generations=None):
    """Apply Gaussian mutation with optional decay, preserving critical game input weights."""
    decay = 1.0
    if generation is not None and max_generations:
        decay = max(0.1, 1 - (generation / max_generations))  # Linear decay

    if random.random() < 0.3:
        mutation_rate *= 3  # Occasional strong mutation

    # Critical inputs (indices related to fruit/bomb status and relative positions)
    # Assuming first layer weights for inputs 3,6,9 (fruit/bomb) are critical
    critical_indices = set()
    input_size = 10  # Known input size from game state
    for i in [3, 6, 9]:  # Fruit/bomb status indices
        start_idx = i * input_size
        end_idx = start_idx + input_size
        critical_indices.update(range(start_idx, end_idx))

    for i in range(len(individual)):
        if random.random() < mutation_rate:
            # Reduce mutation strength for critical weights
            mutation_strength = 0.5 * decay
            if i in critical_indices:
                mutation_strength *= 0.5  # Half the mutation strength for critical weights
            individual[i] += random.gauss(0, mutation_strength)
            individual[i] = max(-1, min(1, individual[i]))  # Clamp to [-1, 1]
    return individual

def genetic_algorithm(individual_size, population_size, fitness_function, target_fitness,
                      generations, elite_rate=0.05, mutation_rate=0.25, num_seeds=12):
    """Genetic algorithm with adaptive mutation, elitism, and tournament selection."""
    population = generate_population(individual_size, population_size)
    
    best_individual = None
    best_fitness = float('-inf')
    elite_size = max(1, int(population_size * elite_rate))
    fitness_history = []
    
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
                # ! ? Apply movement penalty based on score variance
                score_variance = np.var(individual_scores)
                movement_penalty = -2.0 if score_variance < 1.0 else 0.0
                adjusted_score = avg_fitness_score + movement_penalty
                fitness = max(0, adjusted_score)
                fitness_scores.append(fitness)

                if fitness > best_fitness:
                    best_fitness = fitness
                    best_individual = individual.copy()
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
        print(f"Generation {generation + 1}: Best = {best_fitness:.2f}, Avg = {avg_fitness:.2f}, Std = {std_fitness:.2f}")

        # Selection and next generation
        parents = select_parents(population, fitness_scores, elite_size)
        new_population = parents[:elite_size]

        while len(new_population) < population_size:
            parent1, parent2 = random.sample(parents, 2)
            child = crossover(parent1, parent2)
            child = mutate(child, mutation_rate, generation, generations)
            new_population.append(child)

        population = new_population
    
    # Summary of evolution progress
    print("\nEvolution Summary:")
    print(f"Initial Best Fitness (Gen 1): {fitness_history[0][0]:.2f}")
    print(f"Final Best Fitness (Gen {generations}): {best_fitness:.2f}")
    print(f"Initial Avg Fitness (Gen 1): {fitness_history[0][1]:.2f}")
    print(f"Final Avg Fitness (Gen {generations}): {avg_fitness:.2f}")
    
    return best_individual, best_fitness
