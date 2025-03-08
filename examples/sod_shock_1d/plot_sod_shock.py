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

RIEMANN_SOLVER = "hllc"
COORD_SYS = "cartesian_1d"  # "cartesian_1d", "cylindrical_1d", or "spherical_1d"
NUM_TOTAL_CELLS = 200
NUM_GHOST_CELLS_SIDE = 1
NUM_CELLS = NUM_TOTAL_CELLS - 2 * NUM_GHOST_CELLS_SIDE
SOLVER = "godunov_first_order"  # "godunov_first_order" or "random_choice"
IS_PLOT_REFERENCE_SOL = True

RESULT_PATH = Path(__file__).parent / "sod_shock_1d.csv"
SOL_PATH = Path(__file__).parent / "sol_cartesian_1d_0.2.npz" # "sol_cartesian_1d_0.2.npz", "sol_cylindrical_1d_0.2.npz" or "sol_spherical_1d_0.2.npz"

CFL = 0.9
TF = 0.2
TOL = 1e-6  # For the riemann solver

### Sod shock parameters ###
GAMMA = 1.4
RHO_L = 1.0
U_L = 0.0
P_L = 1.0
RHO_R = 0.125
U_R = 0.0
P_R = 0.1
DISCONTINUITY_POS = 0.5
LEFT_BOUNDARY_CONDITION = "transmissive"
RIGHT_BOUNDARY_CONDITION = "transmissive"
DOMAIN = (0.0, 1.0)


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
        mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        density[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(
        mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        velocity[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(
        mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        pressure[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
