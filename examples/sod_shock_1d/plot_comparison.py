"""
Compare the results of sod shock test.

Usage:
    python plot_comparison.py

Author: Ching-Yin Ng
Date: April 2025
"""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

IS_PLOT_REFERENCE_SOL = True
RESULT_GODUNOV_PATH = Path(__file__).parent / "snapshots_godunov/snapshot_0.h5"
RESULT_MUSCL_HANCOCK_PATH = Path(__file__).parent / "snapshots_muscl_hancock/snapshot_0.h5"

MARKER_SIZE = 2


def plot_results(result_path: Path, label: str, color: str, axs) -> None:
    with h5py.File(result_path, "r") as f:
        density = f["fields/density"][()]
        velocity = f["fields/velocity_x"][()]
        pressure = f["fields/pressure"][()]
        x_min = f["parameters/x_min"][()]
        x_max = f["parameters/x_max"][()]

    mid_points = np.linspace(
        x_min / (2.0 * len(density)), x_max - x_min / (2.0 * len(density)), len(density)
    )

    axs[0].plot(
        mid_points,
        density,
        "x-",
        label=label,
        color=color,
        markersize=MARKER_SIZE,
    )
    axs[1].plot(
        mid_points,
        velocity,
        "x-",
        label=label,
        color=color,
        markersize=MARKER_SIZE,
    )
    axs[2].plot(
        mid_points,
        pressure,
        "x-",
        label=label,
        color=color,
        markersize=MARKER_SIZE,
    )


def main() -> None:
    _, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Plot the reference solution
    if IS_PLOT_REFERENCE_SOL:
        with h5py.File(RESULT_GODUNOV_PATH, "r") as f:
            coord_sys = f["parameters/coordinate_system"][()].decode("utf-8")
        sol_path = Path(__file__).parent / f"sol_{coord_sys}_0.2.npz"
        x_sol, rho_sol, u_sol, p_sol = np.load(sol_path).values()
        axs[0].plot(x_sol, rho_sol, "r-", label="Reference")
        axs[1].plot(x_sol, u_sol, "r-", label="Reference")
        axs[2].plot(x_sol, p_sol, "r-", label="Reference")

    # Plot first-order results
    plot_results(RESULT_GODUNOV_PATH, "Godunov", "green", axs)

    # Plot second-order results
    plot_results(RESULT_MUSCL_HANCOCK_PATH, "MUSCL Hancock", "blue", axs)

    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")
    axs[1].legend()

    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")
    axs[2].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()

