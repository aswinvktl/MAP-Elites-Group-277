import torch
import torch.nn as nn
import numpy as np

'''
this is like the brain of ONE SINGLE ant
'''

class Controller(nn.Module):
    """
    The ant's brain.
    Takes 8 joint angles as input outputs 8 target angles
    Has 160 weights total - 8x10 + 10x8 = 160
    """

    # fc1 = first layer of connections
    # fc2 = second layer of connections
    # look at the explanation document in research chat
    def __init__(self):
        super(Controller, self).__init__()
        # FIXED 8 inputs, 10 hidden, 8 outputs = 160 weights
        self.fc1 = nn.Linear(8, 10)
        self.fc2 = nn.Linear(10, 8)

        # torch.tanh makes sure output is between -1 and 1
    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return x  # 8 output angles between -1 and 1


    ''' when controller is created, it is randomly initialised with weights. 
    It gets fitness and position after running in sim using archive. It only takes the list of numbers, not the weights.
    Then archive returns numbers. Tht is just a list of numbers not a brain. set_genome() is called and it takes those 160 numbers in that list,
    and loads them into a new brain. and a new working controller is available. That controller gets mutated and tested again'''

    def get_genome(self, device=torch.device("cpu")):
        """Extract all 160 weights as a flat list of numbers. Before it is weights across two layers and now it is list of numbers"""
        params = []
        for param in self.parameters():
            params.append(param.data.detach().flatten().to(device))
        return torch.cat(params).to(device)

    def set_genome(self, genome):
        """Load 160 numbers back into the network weights"""
        offset = 0
        for param in self.parameters():
            size = param.data.numel()
            chunk = genome[offset:offset + size].reshape(param.data.shape).to(param.device)
            with torch.no_grad():
                param.data.copy_(chunk)
            offset += size

    def mutate(self, mutation_strength=0.05):
        """return a new controller with slightly changed weights"""
        genome = self.get_genome()
        noise = mutation_strength * torch.randn_like(genome)
        new_controller = Controller()
        new_controller.set_genome(genome + noise)
        return new_controller

    @staticmethod
    def random():
        """create a brand new controller with random weights"""
        return Controller()

    @staticmethod
    def crossover(parent1, parent2):
        """blend two parent controllers to make a child"""
        genome1 = parent1.get_genome()
        genome2 = parent2.get_genome()
        alpha = torch.rand(1).item()
        child_genome = alpha * genome1 + (1 - alpha) * genome2
        child = Controller()
        child.set_genome(child_genome)
        return child