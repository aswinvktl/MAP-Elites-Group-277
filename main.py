"""
Main file, just tells every file what to do

ADD PRINT STATEMENTS TO SEE WHAT IS HAPPENING, also TEST CODE
"""

import torch
import csv
from datetime import datetime

from controller import Controller
from archive import Archive
from simulation import Simulation
# from visualisation import something


## these values are set as constants for now, which you can change based on the sim and the results you get
MAX_GENERATIONS = 100
POPULATION_SIZE = 5        # number of controllers evaluated per generation
USE_MOCK = True            # set to *false* when running with isaac Sim. for kip and david, this is me blind coding
METRICS_FILE = "map_elites_metrics.csv"
ARCHIVE_FILE = "archive.json"


# logging

# after every generation, write one row to CSV file
def log_metrics(generation, archive):
    """Write one row to the CSV log."""
    file_exists = False
    try:
        with open(METRICS_FILE, "r") as f:
            file_exists = len(f.readlines()) > 0
    except FileNotFoundError:
        pass
    # create archive file if it doesn't exist
    with open(METRICS_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Generation", "Filled_Cells", "Coverage_%", "Best_Fitness"])
        writer.writerow([
            generation,
            archive.filled_cells(),
            round(archive.coverage() * 100, 2),
            round(archive.best_fitness(), 4),
        ])


# main loop
def main():
    device = torch.device("cpu")

    print("=" * 60)
    print("MAP-Elites Ant Controller Evolution")
    print(f"Mode: {'MOCK (no sim)' if USE_MOCK else 'Isaac Sim'}")
    print(f"Generations: {MAX_GENERATIONS} | Population per gen: {POPULATION_SIZE}")
    print("=" * 60)

    # set up components
    archive = Archive(grid_size=10, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0))
    sim = Simulation(num_envs=POPULATION_SIZE, episode_length=200, use_mock=USE_MOCK)

    # try loading a previous archive if one exists
    archive.load(ARCHIVE_FILE)

    # generation Loop
    # run for 100 times and everything happens once per generation
    for generation in range(1, MAX_GENERATIONS + 1):
        print(f"\n--- Generation {generation}/{MAX_GENERATIONS} ---")
        print(f"Archive: {archive.filled_cells()}/100 cells | Coverage: {archive.coverage():.1%}")

        # create or evolve POPULATION_SIZE controllers
        controllers = []
        genomes = []

        """
        Archive has 2+ controllers Pick two, crossover, then mutate
        Archive has 1 controllerPick one, mutate it
        Archive is emptyCreate a completely random controller
        """

        for _ in range(POPULATION_SIZE):
            g1, g2 = archive.sample_two()

            if g1 is not None and g2 is not None:
                # if two parents exist then crossover then mutate
                parent1 = Controller()
                parent1.set_genome(g1.to(device))
                parent2 = Controller()
                parent2.set_genome(g2.to(device))
                child = Controller.crossover(parent1, parent2)
                child = child.mutate()
            elif g1 is not None:
                # one parent, just mutate
                parent = Controller()
                parent.set_genome(g1.to(device))
                child = parent.mutate()
            else:
                # if archive empty create random controller
                child = Controller.random()

            child = child.to(device)
            genome = child.get_genome(device=device)
            controllers.append(child)
            genomes.append(genome)

        # evaluate all controllers in simulation
        # sends all 5 controllers to the sim at same time. Or parallel computing
        results = sim.evaluate(controllers, device=device)

        # update archive with results
        for genome, (fitness, x, y) in zip(genomes, results):
            inserted = archive.insert(genome, fitness, x, y)
            if inserted:
                print(f"  New elite | cell: {archive.get_cell(x, y)} | energy: {fitness:.4f} | pos: ({x:.2f}, {y:.2f})")

        # log and visualise
        log_metrics(generation, archive)

        # TODO - visualise archive
        # something(archive)


    print("\n" + "=" * 60)
    print("MAP-Elites Complete")
    print(f"Final archive: {archive.filled_cells()}/100 cells filled")
    print(f"Final coverage: {archive.coverage():.1%}")
    print(f"Best fitness (lowest energy): {archive.best_fitness():.4f}")

    archive.save(ARCHIVE_FILE)

    print("\nFiles saved:")
    print("  archive_final.png   — heatmap of archive")
    print("  metrics_final.png   — coverage and fitness graphs")
    print(f"  {ARCHIVE_FILE}         — saved archive data")
    print(f"  {METRICS_FILE}  — raw metrics log")


if __name__ == "__main__":
    main()