import torch
import numpy as np

import argparse
from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--task", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

#side note i removed if statement on my end you guys should readd it but it messed with my stuff when testing

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
        """Set up Isaac Sim environment. Called once when use_mock=False."""
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg
        import gymnasium as gym
        
        
        env_cfg = parse_env_cfg(
            args.task,
            device=args.device,
            num_envs=self.num_envs,
        )
        #trying to force rendering

        self.env = gym.make(args.task, cfg=env_cfg)
        
        obs, _ = self.env.reset()
        #try to force spawn
        print(self.env.action_space)
        
        print(f"[SIM] Isaac Sim environment created with {self.num_envs} envs.")

    def evaluate(self, controllers, device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
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
            
            distance = (final_x**2 + final_y**2)**0.5
            fitness = avg_energy - 0.1 * distance
            #set to -0.5 to encourage forward distance
            
            results.append((fitness, final_x, final_y))

        return results

    def _run_real(self, controllers, device):
        """
        Real Isaac Sim evaluation. Runs all controllers in parallel,
        one controller per environment.
        """
        num_controllers = len(controllers)
        assert num_controllers <= self.num_envs, \
            f"Not enough envs ({self.num_envs}) for controllers ({num_controllers})"

        print(f"  [SIM] Evaluating {num_controllers} controllers in parallel...")

        obs_dict, _ = self.env.reset()
        total_energy = torch.zeros(num_controllers, device=device)

        for step in range(self.episode_length):
            joint_angles = obs_dict["policy"][:num_controllers, 12:20].to(device)

            actions = torch.stack([
                controllers[i](joint_angles[i].unsqueeze(0)).squeeze(0)
                for i in range(num_controllers)
            ])

            if num_controllers < self.num_envs:
                padding = torch.zeros(self.num_envs - num_controllers, 8, device=device)
                actions_padded = torch.cat([actions, padding], dim=0)
            else:
                actions_padded = actions

            obs_dict, _, terminated, truncated, info = self.env.step(actions_padded)
            total_energy += torch.abs(actions).mean(dim=1)

        root_pos = self.env.unwrapped.scene["robot"].data.root_pos_w

        results = []
        for i in range(num_controllers):
            final_x = root_pos[i, 0].item()
            final_y = root_pos[i, 1].item()
            avg_energy = total_energy / self.episode_length
            distance = (final_x**2 + final_y**2)**0.5
            fitness = 10 * distance - 0.1 * avg_energy
            print(fitness)
            print(f"  [SIM] Controller {i+1}: energy={avg_energy:.4f}, pos=({final_x:.2f}, {final_y:.2f})")
            results.append((fitness, final_x, final_y))

        return results