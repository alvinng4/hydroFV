"""
Plot the result of Sedov-Taylor blast test

Usage:
    python plot_sedov_blast_2d.py

Reference:
    J. R. Kamm and F. X. Timmes, On Efficient Generation of Numerically Robust Sedov Solutions, LANL Report,
    No. LA-UR-07-2849 (2007).

Author: Ching-Yin Ng
Date: 2025-03-25
"""

import glob
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np

IS_PLOT_REFERENCE_SOL = True
RESULT_PATH = Path(__file__).parent / "snapshots/"
SNAPSHOT_PATH = sorted(glob.glob(str(RESULT_PATH / "*.h5")))[-1]

MARKER_SIZE = 3

RHO_0 = 1.0
U_0 = 0.0
P_0 = 1e-5


def main() -> None:
    with h5py.File(SNAPSHOT_PATH, "r") as f:
        density = f["fields/density"][()]
        velocity_x = f["fields/velocity_x"][()]
        velocity_y = f["fields/velocity_y"][()]
        pressure = f["fields/pressure"][()]
        x_min = f["parameters/x_min"][()]
        x_max = f["parameters/x_max"][()]
        y_min = f["parameters/y_min"][()]
        y_max = f["parameters/y_max"][()]
        num_cells_x = f["parameters/num_cells_x"][()]
        num_cells_y = f["parameters/num_cells_y"][()]
        num_ghost_cells_side = f["parameters/num_ghost_cells_side"][()]

    total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side
    total_num_cells_y = num_cells_y + 2 * num_ghost_cells_side

    density = density.reshape(total_num_cells_x, total_num_cells_y)
    velocity_x = velocity_x.reshape(total_num_cells_x, total_num_cells_y)
    velocity_y = velocity_y.reshape(total_num_cells_x, total_num_cells_y)
    pressure = pressure.reshape(total_num_cells_x, total_num_cells_y)

    density = density[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]
    velocity_x = velocity_x[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]
    velocity_y = velocity_y[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]
    pressure = pressure[num_ghost_cells_side:-num_ghost_cells_side, num_ghost_cells_side:-num_ghost_cells_side]

    velocity = np.sqrt(velocity_x**2 + velocity_y**2)

    x = np.linspace(x_min, x_max, total_num_cells_x)[num_ghost_cells_side:-num_ghost_cells_side]
    y = np.linspace(y_min, y_max, total_num_cells_y)[num_ghost_cells_side:-num_ghost_cells_side]
    X, Y = np.meshgrid(x, y)
    R = np.sqrt(X**2 + Y**2)

    def radial_average(field):
        r_bins = np.linspace(0, np.max(R), 100)
        r_bin_centers = 0.5 * (r_bins[1:] + r_bins[:-1])
        field_avg = np.zeros_like(r_bin_centers)
        for i in range(len(r_bin_centers)):
            mask = (R >= r_bins[i]) & (R < r_bins[i + 1])
            field_avg[i] = np.mean(field[mask])
        return r_bin_centers, field_avg

    r, density_avg = radial_average(density)
    _, velocity_avg = radial_average(velocity)
    _, pressure_avg = radial_average(pressure)

    _, axs = plt.subplots(1, 3, figsize=(14, 4))

    # Plot the reference solution and the actual solution
    if IS_PLOT_REFERENCE_SOL:
        x_sol, rho_sol, u_sol, p_sol = get_reference_sol((0, np.max(r)))
        axs[0].plot(x_sol, rho_sol, "r-")
        axs[1].plot(x_sol, u_sol, "r-")
        axs[2].plot(x_sol, p_sol, "r-")

    axs[0].plot(r, density_avg, "k.", markersize=MARKER_SIZE)
    axs[0].set_xlabel("Radius")
    axs[0].set_ylabel("Density")

    axs[1].plot(r, velocity_avg, "k.", markersize=MARKER_SIZE)
    axs[1].set_xlabel("Radius")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(r, pressure_avg, "k.", markersize=MARKER_SIZE)
    axs[2].set_xlabel("Radius")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()

def get_reference_sol(
    domain: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the reference solution of the Sedov blast problem at t = 1.0s.

    Parameters
    ----------
    domain : tuple[float, float]
        Domain of the solution.

    Returns
    -------
    x_sol : np.ndarray
        Position solution.
    rho_sol : np.ndarray
        Density solution.
    u_sol : np.ndarray
        Velocity solution.
    p_sol : np.ndarray
        Pressure solution.
    """
    r_shock = 0.75
    rho_shock = 6.0
    u_shock = 0.312500
    p_shock = 0.117188
    e_shock = 0.0488281
    lambda_shock = np.flip(
        [
            1.0,
            0.9998,
            0.9802,
            0.9644,
            0.9476,
            0.9295,
            0.9096,
            0.8725,
            0.8442,
            0.8094,
            0.7629,
            0.7242,
            0.6894,
            0.6390,
            0.5745,
            0.5180,
            0.4748,
            0.4222,
            0.3654,
            0.3000,
            0.2500,
            0.2000,
            0.1500,
            0.1000,
        ]
    )
    f_shock = np.flip(
        [
            1.0,
            0.9996,
            0.9645,
            0.9374,
            0.9097,
            0.8812,
            0.8514,
            0.7999,
            0.7638,
            0.7226,
            0.6720,
            0.6327,
            0.5990,
            0.5521,
            0.4943,
            0.4448,
            0.4074,
            0.3620,
            0.3133,
            0.2572,
            0.2143,
            0.1714,
            0.1286,
            0.0857,
        ]
    )
    g_shock = np.flip(
        [
            1.0,
            0.9972,
            0.7651,
            0.6281,
            0.5161,
            0.4233,
            0.3450,
            0.2427,
            0.1892,
            0.1415,
            0.0974,
            0.0718,
            0.0545,
            0.0362,
            0.0208,
            0.0123,
            0.0079,
            0.0044,
            0.0021,
            0.0008,
            0.0003,
            0.0001,
            0.0000,
            0.0000,
        ]
    )
    h_shock = np.flip(
        [
            1.0,
            0.9984,
            0.8658,
            0.7829,
            0.7122,
            0.6513,
            0.5982,
            0.5266,
            0.4884,
            0.4545,
            0.4241,
            0.4074,
            0.3969,
            0.3867,
            0.3794,
            0.3760,
            0.3746,
            0.3737,
            0.3732,
            0.3730,
            0.3729,
            0.3729,
            0.3729,
            0.3729,
        ]
    )

    x_sol = r_shock * lambda_shock
    rho_sol = rho_shock * g_shock
    u_sol = u_shock * f_shock
    p_sol = p_shock * h_shock

    x_sol = np.append(x_sol, np.linspace(x_sol[-1] + 0.0001, domain[1], 500))
    rho_sol = np.append(rho_sol, np.full(500, RHO_0))
    u_sol = np.append(u_sol, np.full(500, U_0))
    p_sol = np.append(p_sol, np.full(500, P_0))

    return x_sol, rho_sol, u_sol, p_sol


if __name__ == "__main__":
    main()
