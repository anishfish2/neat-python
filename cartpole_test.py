#!/usr/bin/env python3
"""
Test the performance of the updated neat-python package on the CartPole-v1 environment.
Includes visualization of the best network and plotting of fitness over generations.
"""

import os
import pickle
import neat
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import visualize
from datetime import datetime

# Constants
NUM_GENERATIONS = 50
RENDER_BEST = True

def eval_genome(genome, config):
    """
    Evaluates a single genome by running it through the CartPole environment.
    Returns the fitness score (total reward).
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make('CartPole-v1')
    
    fitnesses = []
    for _ in range(3):  # Run 3 episodes for better fitness estimation
        observation, _ = env.reset()
        fitness = 0
        done = False
        truncated = False
        
        while not done and not truncated:
            # Get action from neural network
            action = net.activate(observation)
            action = 1 if action[0] > 0.5 else 0  # Convert to discrete action
            
            # Take action in environment
            observation, reward, done, truncated, _ = env.step(action)
            fitness += reward
            
        fitnesses.append(fitness)
    
    env.close()
    return np.mean(fitnesses)  # Return average fitness across episodes

def eval_best_genome(genome, config):
    """
    Evaluates the best genome with rendering.
    """
    print('\nEvaluating best genome...')
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    env = gym.make('CartPole-v1', render_mode='human')
    
    observation, _ = env.reset()
    fitness = 0
    done = False
    truncated = False
    
    while not done and not truncated:
        # Get action from neural network
        action = net.activate(observation)
        action = 1 if action[0] > 0.5 else 0  # Convert to discrete action
        
        # Take action in environment
        observation, reward, done, truncated, _ = env.step(action)
        fitness += reward
        
    env.close()
    return fitness

def run():
    # Load configuration
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-cartpole')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                        neat.DefaultSpeciesSet, neat.DefaultStagnation,
                        config_path)
    
    # Create the population
    pop = neat.Population(config)
    
    # Add reporters
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))
    
    # Lists to store fitness data
    generation_count = []
    best_fitness = []
    avg_fitness = []
    
    # Evaluation function for each generation
    def eval_genomes(genomes, config):
        for genome_id, genome in genomes:
            genome.fitness = eval_genome(genome, config)
        
        # Store statistics
        generation_count.append(len(generation_count))
        gen_best = max(genome.fitness for _, genome in genomes)
        gen_avg = sum(genome.fitness for _, genome in genomes) / len(genomes)
        best_fitness.append(gen_best)
        avg_fitness.append(gen_avg)
    
    # Run evolution
    winner = pop.run(eval_genomes, NUM_GENERATIONS)
    
    # Save the winner
    with open('winner-cartpole', 'wb') as f:
        pickle.dump(winner, f)
    
    # Plot fitness over generations
    plt.figure(figsize=(10, 6))
    plt.plot(generation_count, best_fitness, 'b-', label='Best Fitness')
    plt.plot(generation_count, avg_fitness, 'r-', label='Average Fitness')
    plt.title('Fitness over Generations')
    plt.xlabel('Generation')
    plt.ylabel('Fitness')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'fitness_plot_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.close()
    
    # Visualize the best network
    visualize.draw_net(config, winner, True, node_names={
        -1: 'Cart Position',
        -2: 'Cart Velocity',
        -3: 'Pole Angle',
        -4: 'Pole Angular Velocity',
        0: 'Action'
    })
    
    # Test the winner
    if RENDER_BEST:
        print(f'\nBest genome:\n{winner}')
        fitness = eval_best_genome(winner, config)
        print(f'Best genome final fitness: {fitness}')

if __name__ == '__main__':
    run() 