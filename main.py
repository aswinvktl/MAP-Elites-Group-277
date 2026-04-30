"""
main file, coordinates everything
"""

import torch
import csv
import os
from pathlib import Path
from datetime import datetime

from controller import Controller
from archive import Archive
from simulation import Simulation, args
from simulation import simulation_app
import visualisation

MAX_GENERATIONS = 50
POPULATION_SIZE = 250
USE_MOCK = False

# saves everything into results/ inside the repo so it works for everyone
REPO_DIR = Path(__file__).parent
RUN_DIR = REPO_DIR / "results" / datetime.now().strftime("run_%Y-%m-%d_%H-%M-%S")
RUN_DIR.mkdir(parents=True, exist_ok=True)

METRICS_FILE = RUN_DIR / "metrics.csv"
ARCHIVE_FILE = RUN_DIR / "archive.json"
VISUALISATION_FILE = RUN_DIR / "visualisation-data" / "visual_data.csv"

# writes one row to the csv after every generation
def log_metrics(generation, archive):
    file_exists = METRICS_FILE.exists()
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
    print(f"  [METRICS] Generation {generation} written to: {os.path.abspath(METRICS_FILE)}")

# to find previous runs and start from there
def get_previous_archived_run(repo_dir, current_run_dir):
    results_dir = repo_dir / "results"

    run_dirs = [
        d for d in results_dir.iterdir()
        if d.is_dir() and d.name.startswith("run_") and d != current_run_dir
    ]

    if not run_dirs:
        return None

    run_dirs.sort()
    latest_run = run_dirs[-1]

    archive_path = latest_run / "archive.json"
    return archive_path if archive_path.exists() else None


def main():
    device = torch.device("cuda")

    print("=" * 60)
    print("MAP-Elites Ant Controller Evolution")
    print(f"Mode: {'MOCK (no sim)' if USE_MOCK else 'Isaac Sim'}")
    print(f"Generations: {MAX_GENERATIONS} | Population per gen: {POPULATION_SIZE}")
    print(f"[PATHS] Run folder:          {os.path.abspath(RUN_DIR)}")
    print(f"[PATHS] Archive:             {os.path.abspath(ARCHIVE_FILE)}")
    print(f"[PATHS] Metrics:             {os.path.abspath(METRICS_FILE)}")
    print(f"[PATHS] Visualisation data:  {os.path.abspath(VISUALISATION_FILE)}")
    print("=" * 60)

    archive = Archive(grid_size=10, x_range=(-5.0, 5.0), y_range=(-5.0, 5.0))
    sim = Simulation(num_envs=args.num_envs, episode_length=200, use_mock=USE_MOCK)

    # load previous archive if there is one, otherwise starts fresh
    prev_archive = get_previous_archived_run(REPO_DIR, RUN_DIR)

    if prev_archive is not None:
        archive.load(prev_archive)
    else:
        print("No previous archive found, starting fresh.")    

    generation = 1

    while simulation_app.is_running() and generation < MAX_GENERATIONS:
        print(f"\n--- Generation {generation}/{MAX_GENERATIONS} ---")
        print(f"Archive: {archive.filled_cells()}/100 cells | Coverage: {archive.coverage():.1%}")

        controllers = []
        genomes = []

        for _ in range(POPULATION_SIZE):
            g1, g2 = archive.sample_two()

            if g1 is not None and g2 is not None:
                # two parents, crossover then mutate
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
                # archive is empty so make a random controller
                child = Controller.random()

            child = child.to(device)
            genome = child.get_genome(device=device)
            controllers.append(child)
            genomes.append(genome)

        print(f"  [SIM] Evaluating {len(controllers)} controllers...")
        results = sim.evaluate(controllers, device=device)
        print(f"  [SIM] Got {len(results)} results")

        # make sure the visualisation folder exists before writing
        VISUALISATION_FILE.parent.mkdir(parents=True, exist_ok=True)

        if not VISUALISATION_FILE.exists():
            with open(VISUALISATION_FILE, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["Generation", "Cell", "Fitness", "X", "Y"])

        with open(VISUALISATION_FILE, "a", newline="") as f:
            writer = csv.writer(f)

            # update archive and write new elites to csv
            for genome, (fitness, x, y) in zip(genomes, results):
                inserted = archive.insert(genome, fitness, x, y)

                if inserted:
                    cell = archive.get_cell(x, y)
                    print(f"  [ELITE] cell: {cell} | fitness: {fitness:.4f} | pos: ({x:.2f}, {y:.2f})")
                    writer.writerow([
                        generation,
                        f"({cell[0]},{cell[1]})",
                        round(fitness, 4),
                        round(x, 2),
                        round(y, 2),
                    ])

        log_metrics(generation, archive)
        generation += 1

    print("\n" + "=" * 60)
    print("MAP-Elites Complete")
    print(f"Final archive: {archive.filled_cells()}/100 cells filled")
    print(f"Final coverage: {archive.coverage():.1%}")
    print(f"Best fitness: {archive.best_fitness():.4f}")

    archive.save(ARCHIVE_FILE)

    print("\n[PATHS] Files saved:")
    print(f"  Archive:            {os.path.abspath(ARCHIVE_FILE)}")
    print(f"  Metrics:            {os.path.abspath(METRICS_FILE)}")
    print(f"  Visualisation data: {os.path.abspath(VISUALISATION_FILE)}")

    print("\n[VIS] Running visualisation...")
    visualisation.main(RUN_DIR)

    simulation_app.close()


if __name__ == "__main__":
    main()