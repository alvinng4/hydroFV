import glob
from pathlib import Path

import h5py
from matplotlib import pyplot as plt
import numpy as np

from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)  # Ensure LaTeX is installed and configured

# Define the simulation folders.
SNAPSHOT_FOLDERS = [
    Path(__file__).parent / "snapshots_godunov_256/",
    Path(__file__).parent / "snapshots_mhm_256/",
    Path(__file__).parent / "snapshots_mhm_512/",
]
REFERENCE_PATH = Path(__file__).parent / "max_y_energy_ref.csv"

def natural_sort_key(s):
    """Sort strings with embedded numbers naturally."""
    return [int(Path(s).stem.strip("snapshot_"))]

def main() -> None:
    labels = [
        "Godunov $256^2$",
        "MHM $256^2$",
        "MHM $512^2$"
    ]

    plt.figure(figsize=(10, 4))
    for folder, label in zip(SNAPSHOT_FOLDERS, labels):
        snapshot_files = sorted(
            glob.glob(str(folder / "snapshot_*.h5")), key=natural_sort_key
        )
        max_y_energy = []
        time_arr = []
        for snapshot_file in snapshot_files:
            with h5py.File(snapshot_file, "r") as f:
                t = f["simulation_status/simulation_time"][()]
                velocity_y = f["fields/velocity_y"][()]
                density = f["fields/density"][()]
                num_cells_x = f["parameters/num_cells_x"][()]
                num_cells_y = f["parameters/num_cells_y"][()]
                num_ghost_cells_side = f["parameters/num_ghost_cells_side"][()]

                velocity_y = velocity_y.reshape(
                    (num_cells_x + 2 * num_ghost_cells_side, num_cells_y + 2 * num_ghost_cells_side)
                )[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]
                density = density.reshape(
                    (num_cells_x + 2 * num_ghost_cells_side, num_cells_y + 2 * num_ghost_cells_side)
                )[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]

                y_energy = 0.5 * density * velocity_y**2
                max_y_energy.append(np.max(y_energy))
                time_arr.append(t)

        # Plot the results.
        plt.semilogy(time_arr, max_y_energy, label=label)


    reference_time_arr, reference_max_y_energy = get_reference()
    plt.semilogy(reference_time_arr, reference_max_y_energy, label="Reference", linestyle="--", color="red")

    
    plt.xlabel("Time")
    plt.ylabel(r"$\mathrm{max}\left(\frac{1}{2} \rho v^2_y \right)$")

    plt.xlim(0, 1.5)

    plt.legend()
    plt.tight_layout()
    plt.show()


def get_reference():
    """Read the reference data from a CSV file."""
    x_arr = []
    y_arr = []
    with open(REFERENCE_PATH, "r") as f:
        for line in f:
            parts = line.split(",")
            x_arr.append(float(parts[0].strip()))
            y_arr.append(float(parts[1].strip()))

    return x_arr, y_arr



if __name__ == "__main__":
    main()