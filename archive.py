import numpy as np
import json
import torch


class Archive:
    """
    The MAP-Elites archive.
    A 10x10 grid. Each cell stores the most energy-efficient from that position
    controller found for that region of (x, y) space.
    """
  
    def __init__(self, grid_size=10, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0)):
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range

        # Each cell stores controller genome, fitness, descriptor
        self.grid = {}
        # this sets up the grid 

    def get_cell(self, x, y):
        """
        this will turn the x and y numbers into their grid postions
        and makes sure that the number is between 0 and 9


        
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        i = int((x - x_min) / (x_max - x_min) * self.grid_size)
        j = int((y - y_min) / (y_max - y_min) * self.grid_size)

        # this will calcuate which cell it woll belong to
        i = max(0, min(self.grid_size - 1, i))
        j = max(0, min(self.grid_size - 1, j))

        return (i, j)

        """
        This is called after every ant evaluation.
         when the cell is empty, it will insert the controller to the archive 
         when the cell is not empty, it replaces the one that used the less energy OR the ELITE 
        """
    def insert(self, genome, fitness, x, y):
        """
        add a result to the archive
        fitness = energy used 
        only replaces existing if new fitness is lower
        Returns True if inserted false if rejected
        """
        cell = self.get_cell(x, y)

        if cell not in self.grid or fitness < self.grid[cell]["fitness"]:
            self.grid[cell] = {
                "genome": genome.detach().cpu().clone(),
                "fitness": fitness,
                "descriptor": (x, y),
            }
            return True
        return False

    def sample(self):
        """this picks a random geonome from what has been saved"""
        if len(self.grid) == 0:
            return None
        cell = np.random.choice(list(self.grid.keys()))
        return self.grid[cell]["genome"]

    def sample_two(self):
        """this will pick 2 differnt genomes to be mixed together"""
        if len(self.grid) < 2:
            return None, None
        cells = list(self.grid.keys())
        chosen = np.random.choice(len(cells), size=2, replace=False)
        g1 = self.grid[cells[chosen[0]]]["genome"]
        g2 = self.grid[cells[chosen[1]]]["genome"]
        return g1, g2
        

      
    # this declares the percentage of how much of the grid has been filled
    def coverage(self):
        
        return len(self.grid) / (self.grid_size * self.grid_size) # percentage worked out here 

    def filled_cells(self):
        #this will count how mancy cells have something inside 
        return len(self.grid)

    """
    loops through the archive and returns the lowest energy value
    this is used to get the best controller
    """
    def best_fitness(self):
        #this will find the lowest fitness value
        if len(self.grid) == 0:
            return 0.0
        return min(data["fitness"] for data in self.grid.values())

    def save(self, filename="archive.json"):
        """save the archive to a JSON file."""
        data = {
            "grid_size": self.grid_size,
            "x_range": self.x_range,
            "y_range": self.y_range,
            "cells": {}
        }
        for cell, entry in self.grid.items():
            data["cells"][str(cell)] = {
                "genome": entry["genome"].numpy().tolist(),
                "fitness": float(entry["fitness"]),
                "descriptor": list(entry["descriptor"]),
            }
        with open(filename, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Archive saved to {filename}")

    def load(self, filename="archive.json"):
        #loads the data from the archive file
        try:
            with open(filename, "r") as f:
                data = json.load(f)
            self.grid_size = data["grid_size"]
            self.x_range = tuple(data["x_range"])
            self.y_range = tuple(data["y_range"])
            self.grid = {}
            for cell_str, entry in data["cells"].items():
                import ast
                cell = tuple(ast.literal_eval(cell_str))
                self.grid[cell] = {
                    "genome": torch.tensor(entry["genome"], dtype=torch.float32),
                    "fitness": float(entry["fitness"]),
                    "descriptor": tuple(entry["descriptor"]),
                }
            print(f"Archive loaded: {len(self.grid)} elites")
            return True
        except FileNotFoundError:
            print("No archive file found, starting fresh.")
            return False
