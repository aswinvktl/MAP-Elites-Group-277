import numpy as np
import json
from control import Controller

class Archive:
    """
    Archive class for MAP-Elites algorithm.
    
    Stores the best controllers (elites) in a 2D grid based on behavior descriptors.
    Each cell in the grid holds a controller, its fitness, and the descriptor that placed it there.
    Only replaces a controller if a new one has higher fitness for the same descriptor cell.
    """
    
    def __init__(self, grid_size=10):
        """
        Initialize the archive with a grid of given size.
        
        Args:
            grid_size (int): Size of the square grid (e.g., 10x10).
        """
        self.grid_size = grid_size
        # Grid to store Controller objects (elites)
        self.grid = np.full((grid_size, grid_size), None, dtype=object)
        # Grid to store fitness values for each cell
        self.fitness_grid = np.full((grid_size, grid_size), -np.inf, dtype=np.float32)
        # Grid to store behavior descriptors for each cell
        self.descriptor_grid = np.full((grid_size, grid_size), None, dtype=object)

    def get_cell(self, descriptor):
        """
        Map a 2D behavior descriptor to a grid cell index.
        
        Descriptors are assumed to be in [0, 1] range and are scaled to grid indices.
        
        Args:
            descriptor (array-like): 2-element array [desc1, desc2].
        
        Returns:
            tuple: (x, y) grid coordinates.
        """
        x = int(np.clip(descriptor[0] * self.grid_size, 0, self.grid_size - 1))
        y = int(np.clip(descriptor[1] * self.grid_size, 0, self.grid_size - 1))
        return x, y

    def add_controller(self, controller, fitness, descriptor):
        """
        Add a controller to the archive if it improves the fitness for its descriptor cell.
        
        Args:
            controller (Controller): The controller to potentially add.
            fitness (float): Fitness score of the controller.
            descriptor (array-like): 2D behavior descriptor.
        
        Returns:
            bool: True if added (improved), False otherwise.
        """
        x, y = self.get_cell(descriptor)
        if fitness > self.fitness_grid[x, y]:
            self.grid[x, y] = controller.copy()
            self.fitness_grid[x, y] = fitness
            self.descriptor_grid[x, y] = np.asarray(descriptor, dtype=np.float32)
            return True
        return False

    def get_random_elite(self):
        """
        Retrieve a random elite controller from the archive.
        
        Used for parent selection in evolution.
        
        Returns:
            Controller or None: A copy of a random elite, or None if archive is empty.
        """
        elites = [(x, y) for x in range(self.grid_size) for y in range(self.grid_size) if self.grid[x, y] is not None]
        if not elites:
            return None
        idx = np.random.choice(len(elites))
        ex, ey = elites[idx]
        return self.grid[ex, ey].copy()

    def get_coverage(self):
        """
        Calculate the percentage of grid cells that have been filled with elites.
        
        Returns:
            float: Coverage ratio (0.0 to 1.0).
        """
        filled_cells = sum(1 for cell in self.grid.flat if cell is not None)
        return filled_cells / (self.grid_size * self.grid_size)

    def filled_cells(self):
        """
        Count the number of filled cells in the grid.
        
        Returns:
            int: Number of cells with elites.
        """
        return sum(1 for cell in self.grid.flat if cell is not None)

    def best_fitness(self):
        """
        Get the highest fitness value in the archive.
        
        Returns:
            float: Best fitness, or 0.0 if archive is empty.
        """
        if self.filled_cells() == 0:
            return 0.0
        return float(np.max(self.fitness_grid[self.fitness_grid > -np.inf]))

    def summary(self):
        """
        Get a summary of the archive's current state.
        
        Returns:
            dict: Summary with grid_size, coverage, filled_cells, best_fitness.
        """
        return {
            "grid_size": self.grid_size,
            "coverage": self.get_coverage(),
            "filled_cells": self.filled_cells(),
            "best_fitness": self.best_fitness(),
        }

    def save_to_file(self, filename="archive.json"):
        """
        Save the archive to a JSON file for persistence.
        
        Includes grid size, fitness grid, and controller data with descriptors.
        
        Args:
            filename (str): Path to save the file.
        """
        data = {
            "grid_size": self.grid_size,
            "fitness_grid": self.fitness_grid.tolist(),
            "controllers": {}
        }
        for x in range(self.grid_size):
            for y in range(self.grid_size):
                if self.grid[x, y] is not None:
                    data["controllers"][f"{x},{y}"] = {
                        "params": self.grid[x, y].params.tolist(),
                        "fitness": float(self.fitness_grid[x, y]),
                        "descriptor": self.descriptor_grid[x, y].tolist() if self.descriptor_grid[x, y] is not None else None,
                    }

        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)

    def load_from_file(self, filename="archive.json"):
        """
        Load the archive from a JSON file.
        
        Reconstructs the grids and controllers from saved data.
        
        Args:
            filename (str): Path to the file to load.
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)

            self.grid_size = data["grid_size"]
            self.fitness_grid = np.array(data["fitness_grid"], dtype=np.float32)
            self.grid = np.full((self.grid_size, self.grid_size), None, dtype=object)
            self.descriptor_grid = np.full((self.grid_size, self.grid_size), None, dtype=object)

            for key, cell in data["controllers"].items():
                x, y = map(int, key.split(','))
                self.grid[x, y] = Controller(cell["params"])
                self.fitness_grid[x, y] = float(cell["fitness"])
                self.descriptor_grid[x, y] = np.array(cell["descriptor"], dtype=np.float32) if cell["descriptor"] is not None else None
        except FileNotFoundError:
            pass  # Archive file doesn't exist, start with empty archive
