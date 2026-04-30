"""
Visualisation.py


"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import csv
import os


def load_data(filename, has_header=False):
    # Initilise parsed
    parsed = []

    # Opens the csv file
    with open(filename, newline="") as f:
        reader = csv.reader(f)
        
        # If the data has a header it will skip that row
        if has_header:
            next(reader, None)

        for row in reader:
            # If there is a row missing or the length of the row is too short then it will bypass that row
            if not row or len(row) < 5:
                continue

            _, cell_str, energy, pos_x, pos_y = row

            # Removes the parentheses ands splits the data
            cell_x, cell_y = map(int, cell_str.strip("()").split(","))

            # Appeneds the data to the parsed variable
            parsed.append((
                cell_x,
                cell_y,
                float(energy),
                float(pos_x),
                float(pos_y)
            ))

    return parsed


def build_grid(parsed):
    # Stops crashes if parsed is empty
    if not parsed:
        raise ValueError("No valid data found in CSV")

    # Chooses the size of the heatmap based on how much is in our data
    max_x = max(p[0] for p in parsed) + 1
    max_y = max(p[1] for p in parsed) + 1

    # Initialise grid and counts
    grid = np.zeros((max_y, max_x))
    counts = np.zeros((max_y, max_x))

    # Accumulates the values per cell and counts how many values each cell gets
    for cell_x, cell_y, energy, *_ in parsed:
        grid[cell_y, cell_x] += energy
        counts[cell_y, cell_x] += 1

    # Calculates the average of the sum of the energies
    with np.errstate(invalid='ignore'):
        grid = np.divide(
            grid,
            counts,
            where=counts != 0,
            out=np.full_like(grid, np.nan)
        )
    
    return grid


def plot_heatmap(grid, filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]

    # Plots out the data onto a heatmap and saves it to a file
    ax = sns.heatmap(grid[::-1], cmap="viridis", cbar_kws={'label': 'Energy'})
    plt.xlabel('Cell X')
    plt.ylabel('Cell Y')
    plt.title('Energy Heatmap')
    plt.savefig(f"visualisation-data/Heatmap_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def plot_scatter_graph(parsed, filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]  

    # Seperates data into lines for plotting
    x = [p[3] for p in parsed]  # pos_x
    y = [p[4] for p in parsed]  # pos_y
    energy = [p[2] for p in parsed]

    # Plots out the data onto a scatter graph and saves it to a file
    plt.scatter(x, y, c=energy, cmap="viridis")
    plt.colorbar(label='Energy')
    plt.xlabel('Pos X')
    plt.ylabel('Pos Y')
    plt.title('Energy Scatter Plot')
    plt.savefig(f"visualisation-data/Scatter_Graph_{name}.png", dpi=300, bbox_inches='tight')
    plt.close()


def main():
    filename = "visualisation-data/TEST_DATA.csv"

    parsed = load_data(filename, has_header=False)
    grid = build_grid(parsed)
    plot_heatmap(grid, filename)
    plot_scatter_graph(parsed, filename)


if __name__ == "__main__":
    main()
