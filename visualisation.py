"""
Visualisation.py

Takes data from visual_data.csv, parses it, builds a grid,
then plots a heatmap and scatter graph into the same run folder.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os
from pathlib import Path

# Loads the data from the specificed file in main, strips the code and parses the data into a new variable
def load_data(filename, has_header=False):
    parsed = []

    with open(filename, newline="") as f:
        reader = csv.reader(f)

        if has_header:
            next(reader, None)

        for row in reader:
            if not row or len(row) < 5:
                continue

            _, cell_str, energy, pos_x, pos_y = row
            cell_x, cell_y = map(int, cell_str.strip("()").split(","))

            parsed.append((
                cell_x,
                cell_y,
                float(energy),
                float(pos_x),
                float(pos_y)
            ))

    return parsed


# Takes the parsed data and builds out grid points for use in the heatmap and graphs
def build_grid(parsed):
    if not parsed:
        raise ValueError("No valid data found in CSV")

    max_x = max(p[0] for p in parsed) + 1
    max_y = max(p[1] for p in parsed) + 1

    grid = np.zeros((max_y, max_x))
    counts = np.zeros((max_y, max_x))

    for cell_x, cell_y, energy, *_ in parsed:
        grid[cell_y, cell_x] += energy
        counts[cell_y, cell_x] += 1

    with np.errstate(invalid='ignore'):
        grid = np.divide(
            grid,
            counts,
            where=counts != 0,
            out=np.full_like(grid, np.nan)
        )

    return grid


# Uses the newly built grid and plots out a heatmap
def plot_heatmap(grid, out_dir, name):
    output_path = out_dir / f"Heatmap_{name}.png"
    sns.heatmap(grid[::-1], cmap="viridis", cbar_kws={'label': 'Energy'})
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Energy Heatmap')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [VIS] Heatmap saved to: {os.path.abspath(output_path)}")


# Uses the newly built grid and plots out a scatter graph
def plot_scatter_graph(parsed, out_dir, name):
    output_path = out_dir / f"Scatter_Graph_{name}.png"

    x = [p[3] for p in parsed]
    y = [p[4] for p in parsed]
    energy = [p[2] for p in parsed]

    plt.scatter(x, y, c=energy, cmap="viridis")
    plt.colorbar(label='Energy')
    plt.xlabel('Pos X')
    plt.ylabel('Pos Y')
    plt.title('Scatter Graph')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [VIS] Scatter graph saved to: {os.path.abspath(output_path)}")


# Main
def main(run_dir=None):
    if run_dir is None:
        # fallback to old behaviour (optional but nice)
        REPO_DIR = Path(__file__).parent
        results_dir = REPO_DIR / "results"

        run_folders = sorted(results_dir.iterdir(), reverse=True)
        if not run_folders:
            print("[VIS] No run folders found in results/")
            return

        run_dir = run_folders[0]

    vis_dir = Path(run_dir) / "visualisation-data"
    filename = vis_dir / "visual_data.csv"

    print(f"[VIS] Reading data from: {os.path.abspath(filename)}")

    parsed = load_data(filename, has_header=True)
    grid = build_grid(parsed)

    plot_heatmap(grid, vis_dir, "visual_data")
    plot_scatter_graph(parsed, vis_dir, "visual_data")

    print(f"[VIS] All visualisations saved to: {os.path.abspath(vis_dir)}")


if __name__ == "__main__":
    main()