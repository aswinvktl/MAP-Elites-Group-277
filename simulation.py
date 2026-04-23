import torch
import numpy as np


class Simulation:
    """
    Handles running the ant in isaac sim and returning results

    For now this uses MOCK data so everyone can test without Isaac Sim
    When David/Kip have confirmed the real x/y position values,
    the run_real() method will be uncommented and used instead
    """

    def __init__(self, num_envs=5, episode_length=200, use_mock=True):
        self.num_envs = num_envs
        self.episode_length = episode_length
        self.use_mock = use_mock

        if not use_mock:
            self._setup_isaac_sim()

    def _setup_isaac_sim(self):
        """
        Set up Isaac Sim environment.
        Only called when use_mock=False.
        David/Kip: fill this in once confirmed working.
        """
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        # These will be passed in from command line args in real use
        # For now placeholder
        raise NotImplementedError("Real sim setup: David/Kip to complete.")

    def evaluate(self, controllers, device=torch.device("cpu")):
        """
        Run a batch of controllers and return results

        Args:
            controllers: list of Controller objects

        Returns:
            list of (fitness, x, y) tuples
            fitness = energy used (lower is better)
            x, y = final position of ant after episode
        """
        if self.use_mock:
            return self._run_mock(controllers, device)
        else:
            return self._run_real(controllers, device)

    def _run_mock(self, controllers, device):
        """
        Fake simulation for testing without Isaac Sim.
        Returns random but plausible (fitness, x, y) per controller.
        """
        results = []

        for controller in controllers:
            total_energy = 0.0
            final_x = 0.0
            final_y = 0.0

            # Simulate episode steps
            for step in range(self.episode_length):
                # Mock observation: 8 random joint angles
                obs = torch.randn(self.num_envs, 8) * 0.5

                with torch.no_grad():
                    action = controller(obs.to(device))

                # Mock energy: average absolute joint velocity
                joint_velocity = torch.abs(action).mean().item()
                total_energy += joint_velocity

                # Mock position: accumulate small movements
                final_x += obs[:, 0].mean().item() * 0.01
                final_y += obs[:, 1].mean().item() * 0.01

            avg_energy = total_energy / self.episode_length
            results.append((avg_energy, final_x, final_y))

        return results

    def _run_real(self, controllers, device):
        """
        Real Isaac Sim evaluation.
        David/Kip: complete this once x/y position confirmed.

        What needs to go here:
        1. Reset all envs
        2. For each step in episode:
             - Get joint positions from obs["policy"][:, 12:20]  (8 joint angles)
             - Pass through controller to get actions
             - Step the environment
             - Accumulate joint velocities for energy
        3. After episode:
             - Get final x, y from root_pos_w
             - Calculate average energy
        4. Return list of (energy, x, y)
        """
        raise NotImplementedError("Real sim: David/Kip to complete.")