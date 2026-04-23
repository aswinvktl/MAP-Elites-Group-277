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
        """Set up Isaac Sim environment. Called once when use_mock=False."""
        import gymnasium as gym
        import isaaclab_tasks
        from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

        env_cfg = parse_env_cfg(
            "Isaac-Ant-v0",
            num_envs=self.num_envs,
        )
        env_cfg.episode_length_s = self.episode_length / 60.0  # assumes 60Hz sim
        self.env = gym.make("Isaac-Ant-v0", cfg=env_cfg)
        print(f"[SIM] Isaac Sim environment created with {self.num_envs} envs.")

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
        Real Isaac Sim evaluation. Runs one controller at a time (each uses all num_envs slots
        for parallel rollouts, averaged to a single result).
        """
        results = []

        for idx, controller in enumerate(controllers):
            print(f"  [SIM] Evaluating controller {idx + 1}/{len(controllers)}...")
            obs_dict, _ = self.env.reset()

            total_energy = 0.0

            for step in range(self.episode_length):
                # Joint angles are at indices 12:20 in the policy observation
                joint_angles = obs_dict["policy"][:, 12:20].to(device)  # shape: (num_envs, 8)

                with torch.no_grad():
                    actions = controller(joint_angles)  # shape: (num_envs, 8)

                obs_dict, _, terminated, truncated, info = self.env.step(actions.cpu().numpy())

                # Energy = mean absolute joint torque/velocity across envs and joints
                total_energy += torch.abs(actions).mean().item()

            avg_energy = total_energy / self.episode_length

            # Final x, y position of the ant (averaged across envs)
            # root_pos_w is the world-frame root position: shape (num_envs, 3)
            root_pos = self.env.unwrapped.scene["robot"].data.root_pos_w  # (num_envs, 3)
            final_x = root_pos[:, 0].mean().item()
            final_y = root_pos[:, 1].mean().item()

            print(f"  [SIM] Controller {idx + 1}: energy={avg_energy:.4f}, pos=({final_x:.2f}, {final_y:.2f})")
            results.append((avg_energy, final_x, final_y))

        return results