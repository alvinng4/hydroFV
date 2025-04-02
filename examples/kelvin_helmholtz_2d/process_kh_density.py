import csv
import glob
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
import numpy as np

# Define the simulation folders.
SNAPSHOT_FOLDERS = [
    Path(__file__).parent / "snapshots_godunov_256/",
    Path(__file__).parent / "snapshots_mhm_256/",
    Path(__file__).parent / "snapshots_mhm_512/",
]

def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(Path(s).stem.strip("snapshot_"))]

def main() -> None:
    # Define the target times to plot.
    target_times = [1.5, 3.0, 9.0, 15.0]
    tolerance = 0.1  # Allow a little slack when matching simulation time

    for folder in SNAPSHOT_FOLDERS:
        snapshot_files = sorted(
            glob.glob(str(folder / "snapshot_*.h5")), key=natural_sort_key
        )
        folder_data = {}
        for snapshot_file in snapshot_files:
            with h5py.File(snapshot_file, "r") as f:
                t = f["simulation_status/simulation_time"][()]
                for tt in target_times:
                    if abs(t - tt) < tolerance:
                        density = f["fields/density"][()]
                        x_min = f["parameters/x_min"][()]
                        x_max = f["parameters/x_max"][()]
                        y_min = f["parameters/y_min"][()]
                        y_max = f["parameters/y_max"][()]
                        num_cells_x = f["parameters/num_cells_x"][()]
                        num_cells_y = f["parameters/num_cells_y"][()]
                        num_ghost_cells_side = f["parameters/num_ghost_cells_side"][()]

                        density = density.reshape(
                            (num_cells_x + 2 * num_ghost_cells_side, num_cells_y + 2 * num_ghost_cells_side)
                        )[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]
                        density = density.flatten()

                        # Write the data to a CSV file.
                        # Create the directory if it doesn't exist
                        output_dir = Path("processed_files/")
                        output_dir.mkdir(parents=True, exist_ok=True)
                        
                        csv_file_path = output_dir / f"{folder.name}_{tt}.csv"

                        with open(csv_file_path, "w", newline="") as csvfile:
                            writer = csv.writer(csvfile)
                            writer.writerow(["x", "y", "density"])
                            for i in range(len(density)):
                                x = x_min + (x_max - x_min) * ((i + num_ghost_cells_side) / (num_cells_x + 2 * num_ghost_cells_side))
                                y = y_min + (y_max - y_min) * ((i + num_ghost_cells_side) / (num_cells_y + 2 * num_ghost_cells_side))
                                writer.writerow([x, y, density[i]])





if __name__ == "__main__":
    main()