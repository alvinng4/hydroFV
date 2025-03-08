"""
Plot the result of sod shock tube test

Usage:
    python plot_sod_shock.py

Author: Ching-Yin Ng
Date: 2025-03-07
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

COORD_SYS = "cartesian_1d"  # "cartesian_1d", "cylindrical_1d", or "spherical_1d"
IS_PLOT_REFERENCE_SOL = True

RESULT_PATH = Path(__file__).parent / "sod_shock_1d.csv"

if COORD_SYS == "cartesian_1d":
    SOL_PATH = Path(__file__).parent / "sol_cartesian_1d_0.2.npz"
elif COORD_SYS == "cylindrical_1d":
    SOL_PATH = Path(__file__).parent / "sol_cylindrical_1d_0.2.npz"
elif COORD_SYS == "spherical_1d":
    SOL_PATH = Path(__file__).parent / "sol_spherical_1d_0.2.npz"


def main() -> None:
    result_df = pl.read_csv(RESULT_PATH)

    mid_points = result_df["mid_point"].to_numpy()
    density = result_df["density"].to_numpy()
    velocity = result_df["velocity"].to_numpy()
    pressure = result_df["pressure"].to_numpy()

    _, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Plot the reference solution and the actual solution
    if IS_PLOT_REFERENCE_SOL:
        x_sol, rho_sol, u_sol, p_sol = np.load(SOL_PATH).values()
        axs[0].plot(x_sol, rho_sol, "r-")
        axs[1].plot(x_sol, u_sol, "r-")
        axs[2].plot(x_sol, p_sol, "r-")

    axs[0].plot(
        mid_points,
        density,
        "k.",
        markersize=2,
    )
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(
        mid_points,
        velocity,
        "k.",
        markersize=2,
    )
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(
        mid_points,
        pressure,
        "k.",
        markersize=2,
    )
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
