import numpy as np
import json
import torch


class Archive:
    """
    The MAP-Elites archive.
    A 10x10 grid. Each cell stores the most energy-efficient
    controller found for that region of (x, y) space.
    """
    # makes the grid. it is 10x10 and 100 cells in total.
    # it covers from -5 to 5 in both x and y directions.
    # these are placeholders for kip and david until you know how far the ant actually goes
    def __init__(self, grid_size=10, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0)):
        self.grid_size = grid_size
        self.x_range = x_range
        self.y_range = y_range

        # Each cell stores controller genome, fitness, descriptor
        self.grid = {}

    def get_cell(self, x, y):
        """Convert a real x, y position into a grid cell index.
        It takes the real position like x = 2.3 y = -1.2 and coverts it to grid cell (2, 1)
        It works by figuring out where the ant falls within a range and converts to a grid index between 0 and 9
        For example, if the ant ends up at x=2.3 and y=-1.1, and the range is -5 to 5:
        x_cell = (2.3 - (-5)) / (5 - (-5)) * 10 = 7.3  ->cell index 7
        y_cell = (-1.1 - (-5)) / (5 - (-5)) * 10 = 3.9 ->cell index 3

        Resulting grid cell: (7, 3)
        """
        x_min, x_max = self.x_range
        y_min, y_max = self.y_range

        i = int((x - x_min) / (x_max - x_min) * self.grid_size)
        j = int((y - y_min) / (y_max - y_min) * self.grid_size)

        # Clamp to grid bounds
        i = max(0, min(self.grid_size - 1, i))
        j = max(0, min(self.grid_size - 1, j))

        return (i, j)

        """
        This is called after every ant evaluation. It does one of the two things,
        - if cell is empty it inserts the controller into the archive
        - if it is not empty, it replaces the one that used the less energy OR the ELITE 
        """
    def insert(self, genome, fitness, x, y):
        """
        add a result to the archive
        fitness = energy used (LOWER is better)
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
        """Pick a random genome from whatever is currently stored"""
        if len(self.grid) == 0:
            return None
        cell = np.random.choice(list(self.grid.keys()))
        return self.grid[cell]["genome"]

    def sample_two(self):
        """Pick two different random genomes for crossover"""
        if len(self.grid) < 2:
            return None, None
        cells = list(self.grid.keys())
        chosen = np.random.choice(len(cells), size=2, replace=False)
        g1 = self.grid[cells[chosen[0]]]["genome"]
        g2 = self.grid[cells[chosen[1]]]["genome"]
        return g1, g2

        """ this is mostly to get data and visualise it.
         It gets the percentage, and this needs to be used in visualisation.py - for seb"""

    def coverage(self):
        """What percentage of the grid is filled"""
        return len(self.grid) / (self.grid_size * self.grid_size)

    def filled_cells(self):
        """How many cells are filled"""
        return len(self.grid)

    """
    loops through the archive and returns the lowest energy value
    this is used to get the best controller
    """
    def best_fitness(self):
        """get the lowest (best) energy value in the archive."""
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
        """load the archive from a file"""
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