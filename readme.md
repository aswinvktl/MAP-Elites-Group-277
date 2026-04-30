## Implementation of Evolutionary Computation algorithm on parallel computing for a quadruped robot

# Overview
Robust control is a requirement for trustworthy autonomous robots. Robot learning is an active research field with state-of-the-art results in autonomous control. In this project, you will implement an Evolutionary Computation algorithm to train a quadruped robot on the Isaac Lab sim. This simulator allows for fast and parallel processing using NVidia GPUs. You will follow algorithms already implemented by the community to implement your own version of MAP-Elites algorithms and will run it on the university's high-performance computing cluster (HPC). The goal is for the robot to learn to navigate complex environments.

Developing the controller in Isaac Lab requires skills in C++ and Python. Essential skills for using the HPC include:

     Linux command line: Getting started - Advanced tutorial 
     Slurm
 
You will implement a container solution, e.g. Docker or Apptainer, that will be deployed to the HPC. The HPC will run the learning algorithms, and you will be able to visualise the learned control on the Isaac Lab sim. If time allows, the controller can be run on the real robot on my research group.

# Deliverables
      A container solution that will be deployed to the HPC.
      A training policy for the MAP-Elites algorithms resulted in a control for the robot.
      A set of diagrams with the learning performance using Tensorboard.
      Code and documentation uploaded to our Git server.

# Resources Available
     Access to the Napier HPC.
     URDF file that defines the robot inside the simulation.
     Community-built algorithms, e.g. stable baselines.
     Access to a physical version of the Hexapod Qutier, implemented following the Qutee robot from Imperial College.

# Contacts
  Client: Simon Smith
  Sponsor: Leni Le Goff

# Requirements
     Run in terminal "pip install -r requirements.txt" to install needed imports