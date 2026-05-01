## Evolutionary Computation for Quadruped Robot Control
# Overview
This project implements an evolutionary computation algorithm (MAP-Elites) to train a quadruped robot in simulation. The system uses parallel computation on GPUs to efficiently explore control strategies.

Training is performed in Isaac Lab, enabling large-scale experimentation through high-performance computing (HPC). The goal is to develop robust policies capable of navigating complex environments.
 
# Key Features
* MAP-Elites evolutionary algorithm implementation
* Parallel training on GPU-enabled HPC systems
* Integration with Isaac Lab simulator

# Resources Used
* Python
* Nividia GPU Computing
* Issac Lab / Issac Sim

# Installation
Clone the repository and install dependencies:

git clone https://github.com/aswinvktl/MAP-Elites-Group-277.git
cd MAP-Elites-Group-277
pip install -r requirements.txt

# Usage
Run main.py t

# Requirements
* Python 3.x
* NVIDIA GPU + CUDA support
* Isaac Lab / Isaac Sim installed

# Results
Picture of the Issac Sim as it was running:
![Screenshot](screenshots\Issac_Sim_Screenshot.png)

# Future Work
* Deploy controller on physical quadruped robot
* Extend algorithm to diverse terrains and conditions
* Improve training efficiency and robustness

# Team
     Aswin Vazhakkoottathil Podimon (Project Manager)
     David Weir
     Jamie Harris
     Sebastian Murray
     Robbie Black
     Kipras Tomkevicius

# Contacts
     Client: Simon Smith
     Sponsor: Leni Le Goff

# License
MIT