import torch
import torch.nn as nn
import random
import numpy as np
import matplotlib.pyplot as plt
import json
import csv
from datetime import datetime

# components for map elites
archive = {}  # stores elites

# Generation and population parameters
max_generations = 3
population_size = 50

# Archive file paths
archive_file = "map_elites_archive.json"
metrics_file = "map_elites_metrics.csv"

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
    
    def get_weights_as_vector(self, device=torch.device("cpu")):
        params = []
        for param in self.parameters():
            params.append(param.data.detach().flatten().to(device))
        return torch.cat(params).to(device)
    
    def set_weights_from_vector(self, vector):
        # vector may be on CPU or GPU; ensure we copy to param.device
        offset = 0
        for param in self.parameters():
            size = param.data.numel()
            chunk = vector[offset:offset + size].reshape(param.data.shape).to(param.device)
            with torch.no_grad():
                param.data.copy_(chunk)
            offset += size


def create_policy(device):
    return SimplePolicy().to(device)


def compute_behavior_descriptor(policy_obs):
    # policy_obs: tensor (num_envs, obs_dim) or (obs_dim,)
    if isinstance(policy_obs, np.ndarray):
        policy_obs = torch.from_numpy(policy_obs).float()
    if policy_obs.dim() == 1:
        policy_obs = policy_obs.unsqueeze(0)
    # forward velocity = x velocity (index 0)
    forward_velocity = policy_obs[:, 0]  # per-env
    # energy = sum abs joint velocities (example indices 24:36)
    if policy_obs.shape[1] >= 36:
        energy = torch.sum(torch.abs(policy_obs[:, 24:36]), dim=1)
    else:
        energy = torch.sum(torch.abs(policy_obs), dim=1)
    # Return averaged descriptor over envs
    return torch.tensor([forward_velocity.mean().item(), energy.mean().item()])


def compute_fitness(total_reward):
    # total_reward: scalar tensor or float (we assume scalar mean reward across envs)
    if isinstance(total_reward, torch.Tensor):
        return float(total_reward.item())
    return float(total_reward)


def get_cell(descriptor):
    # descriptor values may be unbounded; we normalize via simple clipping + scaling
    # expected forward_velocity range [-1, 1] -> map to 0..9
    # energy range >=0 -> we clip to a reasonable max (e.g., 10) then map to 0..9
    fv = float(descriptor[0])
    en = float(descriptor[1])

    fv_norm = max(-1.0, min(1.0, fv))  # clamp
    fv_idx = int((fv_norm + 1.0) / 2.0 * 9.0)  # map -1..1 -> 0..9

    en_max = 10.0
    en_clamped = max(0.0, min(en_max, en))
    en_idx = int(en_clamped / en_max * 9.0)

    fv_idx = max(0, min(9, fv_idx))
    en_idx = max(0, min(9, en_idx))
    return (fv_idx, en_idx)


def mutate(genome):
    device = genome.device if isinstance(genome, torch.Tensor) else torch.device("cpu")
    noise = 0.05 * torch.randn_like(genome, device=device)
    print(f"  [MUTATE] Original genome mean: {genome.mean().item():.6f}, std: {genome.std().item():.6f}")
    mutated = (genome.to(device) + noise).to(genome.device)
    print(f"  [MUTATE] Mutated genome mean: {mutated.mean().item():.6f}, std: {mutated.std().item():.6f}")
    print(f"  [MUTATE] Mutation noise mean: {noise.mean().item():.6f}, magnitude: {noise.norm().item():.6f}")
    return mutated


def crossover(genome1, genome2):
    if genome2 is None:
        return genome1.clone()
    device = genome1.device
    alpha = random.random()
    result = (alpha * genome1.to(device) + (1 - alpha) * genome2.to(device)).to(genome1.device)
    print(f"  [CROSSOVER] Alpha: {alpha:.3f}, Genome1 mean: {genome1.mean().item():.6f}, Genome2 mean: {genome2.mean().item():.6f}, Result mean: {result.mean().item():.6f}")
    return result


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
    plt.xlabel('Forward Velocity (binned)')
    plt.ylabel('Energy (binned)')
    plt.savefig('archive_test.png')
    plt.close()
    print("  Archive visualization saved to archive_test.png")


def sample_parent():
    if len(archive) == 0:
        return None
    return random.choice(list(archive.values()))["genome"]


def sample_two_parents():
    if len(archive) < 2:
        return None, None
    parents = random.sample(list(archive.values()), 2)
    return parents[0]["genome"], parents[1]["genome"]


def get_archive_coverage():
    return len(archive) / 100.0  # 10x10 grid


def save_archive(filename=None):
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
    if filename is None:
        filename = archive_file
    global archive
    try:
        with open(filename, 'r') as f:
            archive_data = json.load(f)
        archive = {}
        for cell_str, data in archive_data.items():
            import ast
            cell = tuple(ast.literal_eval(cell_str))
            archive[cell] = {
                "genome": torch.tensor(data["genome"], dtype=torch.float32),
                "fitness": float(data["fitness"]),
                "descriptor": torch.tensor(data["descriptor"], dtype=torch.float32),
            }
        print(f"Archive loaded from {filename} ({len(archive)} elites)")
        return True
    except FileNotFoundError:
        print(f"No archive file found at {filename}")
        return False


def log_metrics(generation, archive_size, coverage, best_fitness):
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


def generate_mock_observations(num_envs=8):
    """Generate mock observation data for testing"""
    # Create random observation with forward velocity (idx 0) and joint velocities (idx 24:36)
    obs = np.random.randn(num_envs, 48) * 0.5
    return obs


def main():
    device = torch.device("cpu")
    
    print(f"Device: {device}")
    print(f"Running MAP-Elites with MOCK data (no Isaac Lab required)")
    print("=" * 60)

    num_envs = 8
    action_dim = 12
    episode_length = 200
    generation = 0

    load_archive()

    while generation < max_generations:
        print(f"\n--- Generation {generation + 1}/{max_generations} ---")
        coverage = get_archive_coverage()
        print(f"Archive size: {len(archive)}/100 | Coverage: {coverage:.1%}")

        for pop_idx in range(population_size):
            # Select or create genome
            if random.random() < 0.5 and len(archive) >= 2:
                parent1, parent2 = sample_two_parents()
                if parent1 is not None and parent2 is not None:
                    # parents stored on CPU; move to device for ops then back to CPU
                    p1 = parent1.to(device)
                    p2 = parent2.to(device)
                    genome = crossover(p1, p2)
                    genome = mutate(genome)
                else:
                    parent = sample_parent()
                    if parent is None:
                        policy = create_policy(device)
                        genome = policy.get_weights_as_vector(device=device)
                        print(f"  [CREATE] New random genome")
                    else:
                        genome = mutate(parent.to(device))
                        print(f"  [MUTATE] From parent")
            else:
                parent = sample_parent()
                if parent is None:
                    policy = create_policy(device)
                    genome = policy.get_weights_as_vector(device=device)
                    print(f"  [CREATE] New random genome")
                else:
                    genome = mutate(parent.to(device))
                    print(f"  [MUTATE] From parent")

            # Ensure genome is on device for set_weights
            genome = genome.to(device)
            policy = create_policy(device)
            policy.set_weights_from_vector(genome)

            # SIMULATE: Create mock observations
            # Accumulate rewards over episode (simulated)
            total_reward = 0.0
            final_obs = None
            
            for step_idx in range(episode_length):
                # Generate mock observations
                obs_mock = generate_mock_observations(num_envs)
                obs_tensor = torch.from_numpy(obs_mock).float().to(device)
                
                with torch.no_grad():
                    action = policy(obs_tensor)
                
                # Simulate reward: higher forward velocity and lower energy = higher reward
                forward_vel = obs_mock[:, 0].mean()
                energy = np.abs(obs_mock[:, 24:36]).sum(axis=1).mean()
                step_reward = 1.0 + forward_vel - 0.1 * energy  # bonus for staying alive + forward velocity - energy penalty
                total_reward += step_reward
                
                final_obs = obs_mock

            # Compute descriptor and fitness using final observations
            descriptor = compute_behavior_descriptor(final_obs)
            fitness = compute_fitness(total_reward / episode_length)  # average per step

            cell = get_cell(descriptor)

            # store genome in CPU to keep archive small (and consistent)
            genome_cpu = genome.detach().cpu().clone()

            if cell not in archive or fitness > archive[cell]["fitness"]:
                archive[cell] = {
                    "genome": genome_cpu,
                    "fitness": fitness,
                    "descriptor": descriptor.clone().cpu(),
                }
                print(f"    → New elite in cell {cell} | fitness: {fitness:.3f} | descriptor: ({descriptor[0].item():.3f}, {descriptor[1].item():.3f})")

        # Visualize every generation (small runs)
        visualize_archive(archive)
        best_fitness = max([data["fitness"] for data in archive.values()]) if archive else 0
        log_metrics(generation + 1, len(archive), get_archive_coverage(), best_fitness)

        generation += 1

    print(f"\n{'='*60}")
    print(f"=== MAP-Elites Complete ===")
    print(f"Total generations: {generation}")
    print(f"Final archive size: {len(archive)}/100")
    print(f"Final coverage: {get_archive_coverage():.1%}")

    save_archive()
    visualize_archive(archive)
    print("Final visualization and archive saved")


if __name__ == "__main__":
    main()
