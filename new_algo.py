import argparse
import torch
import sys

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=8, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()


# Launch Isaac Sim
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# Imports AFTER simulator start otherwise it doesnt run

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg


# Algorithm placeholder

def my_algorithm(observation, num_envs, device):
    #action dim is action values per robot this quodroped has 12 joints total so 12 actions
    action_dim = 12
    
    return .15 * torch.randn((num_envs,action_dim),device=device)

# Main Simulation loop

def main():
    
    #build a task config
    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        )
    
    # Create task environment
    env = gym.make(args.task, cfg=env_cfg)

    # Reset once
    obs, info = env.reset()

    # Get device from env
    device = env.unwrapped.device

    print(f"Task environment created: {args.task}")
    print(f"Number of envs: {args.num_envs}")
    
    
    print(f"Observation type: {type(obs)}")
    print(f"Observation keys: {obs.keys()}")
    for key, value in obs.items():
        print(key, value.shape)
        

    while simulation_app.is_running():
        policy_obs = obs["policy"]
        robot0 = policy_obs[0]
        print("base_lin_vel     :", robot0[0:3])
        print("base_ang_vel     :", robot0[3:6])
        print("projected_gravity:", robot0[6:9])
        print("command          :", robot0[9:12])
        print("joint_pos_offset :", robot0[12:24])
        print("joint_vel        :", robot0[24:36])
        print("previous_action  :", robot0[36:48])
        action = my_algorithm(policy_obs, args.num_envs, device)

        # step task env
        obs, reward, terminated, truncated, info = env.step(action)

    env.close()



if __name__ == "__main__":
    main()
    simulation_app.close()