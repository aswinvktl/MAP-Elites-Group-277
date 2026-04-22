import argparse
import torch
import torch.nn as nn
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# 
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


#compoments for map elites 

archive = {}  # stores elites
#counter = 0

# Generation and population parameters
max_generations = 3
population_size = 50

# Archive file paths
archive_file = "map_elites_archive.json"
metrics_file = f"map_elites_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

# Neural network policy
class SimplePolicy(nn.Module):
    def __init__(self, input_size=48, hidden_size=64, output_size=12):
        super(SimplePolicy, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x
    
    def get_weights_as_vector(self):
        """Get all weights as a 1D vector"""
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)
    
    def set_weights_from_vector(self, vector):
        offset = 0
        for param in self.parameters():
            size = param.data.numel()
            chunk = vector[offset:offset + size].reshape(param.data.shape)
            param.data.copy_(chunk.to(param.device))
            offset += size


def create_policy(device):
    """Create a new policy network"""
    return SimplePolicy().to(device)


def compute_behavior_descriptor(obs):
    robot = obs[0]

    forward_velocity = robot[0]  # x velocity
    energy = torch.sum(torch.abs(robot[24:36]))  # joint velocity magnitude

    return torch.tensor([forward_velocity, energy])


def compute_fitness(total_reward): # this will declare the fitness function
    return total_reward.item()


def get_cell(descriptor):
    x = int(max(0, min(9, descriptor[0].item() * 10)))
    y = int(max(0, min(9, descriptor[1].item() * 10)))
    return (x, y)


def mutate(genome):
    return genome + 0.05 * torch.randn_like(genome)


def crossover(genome1, genome2):
    """Blend two parent genomes"""
    alpha = random.random()
    return alpha * genome1 + (1 - alpha) * genome2


def visualize_archive(archive):
    grid = [[0 for _ in range(10)] for _ in range(10)]
    for cell, data in archive.items():
        x, y = cell
        if 0 <= x < 10 and 0 <= y < 10:
            grid[y][x] = data['fitness']
    plt.figure(figsize=(8, 6))
    plt.imshow(grid, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
    plt.colorbar(label='Fitness')
    plt.title('MAP-Elites Archive')
    plt.xlabel('Forward Velocity')
    plt.ylabel('Energy')
    plt.savefig('archive.png')
    plt.close()


def sample_parent():
    if len(archive) == 0:
        return None
    return random.choice(list(archive.values()))["genome"]


def sample_two_parents():
    """Sample two different parents for crossover"""
    if len(archive) < 2:
        return None, None
    parents = random.sample(list(archive.values()), min(2, len(archive)))
    if len(parents) == 2:
        return parents[0]["genome"], parents[1]["genome"]
    return parents[0]["genome"], None


def get_archive_coverage():
    """Calculate archive coverage percentage"""
    return len(archive) / 100.0  # 10x10 grid


def save_archive(filename=None):
    """Save archive to JSON file"""
    if filename is None:
        filename = archive_file
    
    archive_data = {}
    for cell, data in archive.items():
        archive_data[str(cell)] = {
            "genome": data["genome"].cpu().numpy().tolist(),
            "fitness": float(data["fitness"]),
            "descriptor": data["descriptor"].cpu().numpy().tolist(),
        }
    
    with open(filename, 'w') as f:
        json.dump(archive_data, f, indent=2)
    print(f"Archive saved to {filename}")


def load_archive(filename=None):
    """Load archive from JSON file"""
    if filename is None:
        filename = archive_file
    
    global archive
    try:
        with open(filename, 'r') as f:
            archive_data = json.load(f)
        
        archive = {}
        for cell_str, data in archive_data.items():
            cell = tuple(map(int, cell_str.strip("()").split(", ")))
            archive[cell] = {
                "genome": torch.tensor(data["genome"]),
                "fitness": data["fitness"],
                "descriptor": torch.tensor(data["descriptor"]),
            }
        print(f"Archive loaded from {filename} ({len(archive)} elites)")
        return True
    except FileNotFoundError:
        print(f"No archive file found at {filename}")
        return False


def log_metrics(generation, archive_size, coverage, best_fitness):
    """Log generation metrics to CSV"""
    file_exists = False
    try:
        with open(metrics_file, 'r') as f:
            file_exists = len(f.readlines()) > 0
    except FileNotFoundError:
        pass
    
    with open(metrics_file, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Generation', 'Archive_Size', 'Coverage_%', 'Best_Fitness'])
        writer.writerow([generation, archive_size, coverage * 100, best_fitness])

# main loop

def main():

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
    )

    env = gym.make(args.task, cfg=env_cfg)

    device = env.unwrapped.device

    print(f"Task: {args.task}")
    print(f"Num envs: {args.num_envs}")

    action_dim = 12
    episode_length = 200
    generation = 0

    # Try to load previous archive
    load_archive()

    # Main generation loop
    while simulation_app.is_running() and generation < max_generations:
        print(f"\n--- Generation {generation + 1}/{max_generations} ---")
        coverage = get_archive_coverage()
        print(f"Archive size: {len(archive)}/100 | Coverage: {coverage:.1%}")
        
        # Evaluate population
        for pop_idx in range(population_size):
            # Create or evolve a genome (neural network weights)
            if random.random() < 0.5 and len(archive) >= 2:
                # Crossover: blend two parents
                parent1, parent2 = sample_two_parents()
                if parent1 is not None and parent2 is not None:
                    genome = crossover(parent1, parent2)
                    genome = mutate(genome)  # Then mutate
                else:
                    # Fallback to simple mutation
                    parent = sample_parent()
                    if parent is None:
                        policy = create_policy(device)
                        genome = policy.get_weights_as_vector()
                    else:
                        genome = mutate(parent)
            else:
                # Mutation only
                parent = sample_parent()
                if parent is None:
                    policy = create_policy(device)
                    genome = policy.get_weights_as_vector()
                else:
                    genome = mutate(parent)
            
            # Create policy from genome and evaluate
            policy = create_policy(device)
            policy.set_weights_from_vector(genome)

            # resets 
            obs, _ = env.reset()

            total_reward = torch.zeros(args.num_envs, device=device)

            # runs an episode 
            for step_idx in range(episode_length):
                # Get action from policy network
                obs_tensor = torch.from_numpy(obs["policy"]).float().to(device) if isinstance(obs["policy"], np.ndarray) else obs["policy"].to(device)
                with torch.no_grad():
                    action = policy(obs_tensor)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                total_reward += reward

            # Compute descriptor and fitness
            policy_obs = obs["policy"]
            descriptor = compute_behavior_descriptor(policy_obs)
            fitness = compute_fitness(total_reward.mean())

            cell = get_cell(descriptor)

            # Update the archive 
            if cell not in archive or fitness > archive[cell]["fitness"]:
                archive[cell] = {
                    "genome": genome.clone().cpu(),
                    "fitness": fitness,
                    "descriptor": descriptor,
                }

                print(f"New elite in cell {cell} | fitness: {fitness:.3f}")
                #counter += 1
        
        # Visualize and log after each generation
        if (generation + 1) % 10 == 0:
            visualize_archive(archive)
            print(f"Generation {generation + 1}: Saved visualization")
        
        # Log metrics
        best_fitness = max([data["fitness"] for data in archive.values()]) if archive else 0
        log_metrics(generation + 1, len(archive), get_archive_coverage(), best_fitness)
        
        generation += 1
        print("generation")
    
    # Final report and save
    print(f"\n=== MAP-Elites Complete ===")
    print(f"Total generations: {generation}")
    print(f"Final archive size: {len(archive)}/100")
    print(f"Final coverage: {get_archive_coverage():.1%}")
    
    # Save archive and final visualization
    save_archive()
    visualize_archive(archive)
    print(f"Final visualization and archive saved")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()