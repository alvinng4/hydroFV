"""
Plot the result of sod shock tube test

Usage:
    python plot_sod_shock.py

Author: Ching-Yin Ng
Date: 2025-03-11
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

IS_PLOT_REFERENCE_SOL = True
RESULT_PATH = Path(__file__).parent / "snapshots/snapshot_0.h5"

MARKER_SIZE = 4


def main() -> None:
    with h5py.File(RESULT_PATH, "r") as f:
        coord_sys = f["parameters/coordinate_system"][()].decode("utf-8")
        density = f["fields/density"][()]
        velocity = f["fields/velocity_x"][()]
        pressure = f["fields/pressure"][()]
        x_min = f["parameters/x_min"][()]
        x_max = f["parameters/x_max"][()]

    mid_points = np.linspace(
        x_min / (2.0 * len(density)), x_max - x_min / (2.0 * len(density)), len(density)
    )

    _, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Plot the reference solution and the actual solution
    if IS_PLOT_REFERENCE_SOL:
        sol_path = Path(__file__).parent / f"sol_{coord_sys}_0.2.npz"
        x_sol, rho_sol, u_sol, p_sol = np.load(sol_path).values()
        axs[0].plot(x_sol, rho_sol, "r-")
        axs[1].plot(x_sol, u_sol, "r-")
        axs[2].plot(x_sol, p_sol, "r-")

    axs[0].plot(
        mid_points,
        density,
        "k.",
        markersize=MARKER_SIZE,
    )
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(
        mid_points,
        velocity,
        "k.",
        markersize=MARKER_SIZE,
    )
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(
        mid_points,
        pressure,
        "k.",
        markersize=MARKER_SIZE,
    )
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
