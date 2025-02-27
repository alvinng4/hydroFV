"""
Sedov-Taylor blast test

Usage:
    python sedov_blast.py

Reference:
    J. R. Kamm and F. X. Timmes, On Efficient Generation of Numerically Robust Sedov Solutions, LANL Report,
    No. LA-UR-07-2849 (2007).

Author: Ching-Yin Ng
Date: 2025-2-27
"""

import sys
import timeit
import warnings
from pathlib import Path

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import rich.progress

import FiniteVolume1D

RIEMANN_SOLVER = "hllc"
COORD_SYS = "cylindrical_1d"  # "cartesian_1d", "cylindrical_1d", or "spherical_1d"
NUM_TOTAL_CELLS = 512
NUM_GHOST_CELLS_SIDE = 1
NUM_CELLS = NUM_TOTAL_CELLS - 2 * NUM_GHOST_CELLS_SIDE
SOLVER = "godunov_first_order"  # "godunov_first_order" or "random_choice"
IS_PLOT_REFERENCE_SOL = True

CFL = 0.4
TF = 1.0
TOL = 1e-8  # For the riemann solver

### Sedov blast wave parameters ###
NUM_EXPLOSION_CELLS = 1

GAMMA = 1.4
RHO_0 = 1.0
U_0 = 0.0
P_0 = 1e-5
if COORD_SYS == "cartesian_1d":
    E_0 = 0.0673185 / NUM_EXPLOSION_CELLS
elif COORD_SYS == "cylindrical_1d":
    E_0 = 0.311357 / NUM_EXPLOSION_CELLS
elif COORD_SYS == "spherical_1d":
    E_0 = 0.851072 / NUM_EXPLOSION_CELLS
LEFT_BOUNDARY_CONDITION = "reflective"
RIGHT_BOUNDARY_CONDITION = "reflective"
DOMAIN = (0.0, 1.2)


def main() -> None:
    global RIEMANN_SOLVER
    global CFL
    assert COORD_SYS in ["cartesian_1d", "cylindrical_1d", "spherical_1d"]

    c_lib = FiniteVolume1D.utils.load_c_lib()
    system = get_initial_system(NUM_CELLS, COORD_SYS)

    if SOLVER == "random_choice":
        rng = FiniteVolume1D.random_choice.VanDerCorputSequenceGenerator()

        if RIEMANN_SOLVER != "exact":
            msg = "Only exact Riemann solver is supported for random_choice solver. Switching to exact Riemann solver."
            warnings.warn(msg)
            RIEMANN_SOLVER = "exact"

        if CFL > 0.5:
            msg = "The Courant number should be less than 0.5 for random_choice solver. Switching to 0.4."
            warnings.warn(msg)
            CFL = 0.4

    t = 0.0
    num_steps = 0
    start = timeit.default_timer()
    with rich.progress.Progress() as progress:
        print("Simulation in progress...")
        task = progress.add_task("", total=TF)
        while t < TF:
            if num_steps <= 50:
                dt = FiniteVolume1D.utils.get_time_step(CFL * 0.2, system)
            else:
                dt = FiniteVolume1D.utils.get_time_step(CFL, system)

            if t + dt > TF:
                dt = TF - t

            if SOLVER == "godunov_first_order":
                FiniteVolume1D.godunov_first_order.solving_step(
                    c_lib, system, dt, TOL, RIEMANN_SOLVER
                )
            elif SOLVER == "random_choice":
                FiniteVolume1D.random_choice.solving_step(
                    c_lib, system, dt, TOL, RIEMANN_SOLVER, rng
                )

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    end = timeit.default_timer()
    print(f"Done! Num steps: {num_steps}, Time: {end - start:.3f}s")
    print()

    _, axs = plt.subplots(2, 2, figsize=(8, 8))

    # Plot the reference solution and the actual solution
    if IS_PLOT_REFERENCE_SOL:
        x_sol, rho_sol, u_sol, p_sol = get_reference_sol()
        axs[0, 0].plot(x_sol, rho_sol, "r-")
        axs[0, 1].plot(x_sol, u_sol, "r-")
        axs[1, 0].plot(x_sol, p_sol, "r-")

    index_slice = slice(NUM_GHOST_CELLS_SIDE, NUM_TOTAL_CELLS - NUM_GHOST_CELLS_SIDE)

    axs[0, 0].plot(
        system.mid_points[index_slice], system.density[index_slice], "k.", markersize=1
    )
    axs[0, 0].set_xlabel("Position")
    axs[0, 0].set_ylabel("Density")

    axs[0, 1].plot(
        system.mid_points[index_slice], system.velocity[index_slice], "k.", markersize=1
    )
    axs[0, 1].set_xlabel("Position")
    axs[0, 1].set_ylabel("Velocity")

    axs[1, 0].plot(
        system.mid_points[index_slice], system.pressure[index_slice], "k.", markersize=1
    )
    axs[1, 0].set_xlabel("Position")
    axs[1, 0].set_ylabel("Pressure")

    # internal_energy = system.energy - (0.5 * system.density * system.velocity ** 2) * system.volume
    # axs[1, 1].plot(system.mid_points[index_slice], internal_energy[index_slice], "k.", markersize=2)
    # axs[1, 1].set_xlabel("Position")
    # axs[1, 1].set_ylabel("Internal energy")

    plt.tight_layout()
    plt.show()


def get_initial_system(num_cells: int, coord_sys: str) -> FiniteVolume1D.system.System:
    system = FiniteVolume1D.system.System(
        num_cells=num_cells,
        gamma=GAMMA,
        coord_sys=coord_sys,
        domain=DOMAIN,
        num_ghost_cells_side=NUM_GHOST_CELLS_SIDE,
        left_boundary_condition=LEFT_BOUNDARY_CONDITION,
        right_boundary_condition=RIGHT_BOUNDARY_CONDITION,
    )
    system.density.fill(RHO_0)
    system.velocity.fill(U_0)
    system.pressure.fill(P_0)

    # nu = system.alpha + 1.0
    # delta_r = (1.0 / num_cells) * NUM_EXPLOSION_CELLS
    # for i in range(NUM_GHOST_CELLS_SIDE, NUM_GHOST_CELLS_SIDE + NUM_EXPLOSION_CELLS):
    #     system.pressure[i] = (3.0 * (GAMMA - 1.0) * EPS) / ((nu + 1.0) * np.pi * delta_r ** nu)

    system.set_boundary_condition()
    system.convert_primitive_to_conserved()
    system.energy[
        NUM_GHOST_CELLS_SIDE : (NUM_GHOST_CELLS_SIDE + NUM_EXPLOSION_CELLS)
    ] = E_0
    system.convert_conserved_to_primitive()
    system.set_boundary_condition()

    return system


def get_reference_sol() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the reference solution of the Sedov blast problem at t = 1.0s.

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
    if COORD_SYS == "cartesian_1d":
        r_shock = 0.5
        rho_shock = 6.0
        u_shock = 0.277778
        p_shock = 0.0925926
        e_shock = 0.0385802
        lambda_shock = np.flip(
            [
                1.0,
                0.9797,
                0.9420,
                0.9013,
                0.8565,
                0.8050,
                0.7419,
                0.7029,
                0.6553,
                0.5925,
                0.5396,
                0.4912,
                0.4589,
                0.4161,
                0.3480,
                0.2810,
                0.2320,
                0.1680,
                0.1040,
            ]
        )
        f_shock = np.flip(
            [
                1.0,
                0.9699,
                0.9157,
                0.8598,
                0.8017,
                0.7390,
                0.6677,
                0.6263,
                0.5780,
                0.5173,
                0.4682,
                0.4244,
                0.3957,
                0.3580,
                0.2988,
                0.2410,
                0.1989,
                0.1440,
                0.0891,
            ]
        )
        g_shock = np.flip(
            [
                1.0,
                0.8620,
                0.6662,
                0.5159,
                0.3981,
                0.3020,
                0.2201,
                0.1823,
                0.1453,
                0.1075,
                0.0826,
                0.0641,
                0.0535,
                0.0415,
                0.0263,
                0.0153,
                0.0095,
                0.0042,
                0.0013,
            ]
        )
        h_shock = np.flip(
            [
                1.0,
                0.9159,
                0.7917,
                0.6922,
                0.6119,
                0.5458,
                0.4905,
                0.4661,
                0.4437,
                0.4230,
                0.4112,
                0.4037,
                0.4001,
                0.3964,
                0.3929,
                0.3911,
                0.3905,
                0.3901,
                0.3900,
            ]
        )

    elif COORD_SYS == "cylindrical_1d":
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
    elif COORD_SYS == "spherical_1d":
        r_shock = 1.0
        rho_shock = 6.0
        u_shock = 0.333334
        p_shock = 0.117188
        e_shock = 0.0555559
        lambda_shock = np.flip(
            [
                1.0,
                0.9913,
                0.9773,
                0.9622,
                0.9342,
                0.9080,
                0.8747,
                0.8359,
                0.7950,
                0.7493,
                0.6788,
                0.5794,
                0.4560,
                0.3600,
                0.2960,
                0.2000,
                0.1040,
            ]
        )
        f_shock = np.flip(
            [
                1.0,
                0.9814,
                0.9529,
                0.9238,
                0.8745,
                0.8335,
                0.7872,
                0.7398,
                0.6952,
                0.6497,
                0.5844,
                0.4971,
                0.3909,
                0.3086,
                0.2537,
                0.1714,
                0.0891,
            ]
        )
        g_shock = np.flip(
            [
                1.0,
                0.8388,
                0.6454,
                0.4984,
                0.3248,
                0.2275,
                0.1508,
                0.0968,
                0.0620,
                0.0379,
                0.0174,
                0.0052,
                0.0009,
                0.0001,
                0.0000,
                0.0000,
                0.0000,
            ]
        )
        h_shock = np.flip(
            [
                1.0,
                0.9116,
                0.7992,
                0.7082,
                0.5929,
                0.5238,
                0.4674,
                0.4273,
                0.4021,
                0.3857,
                0.3732,
                0.3672,
                0.3656,
                0.3655,
                0.3655,
                0.3655,
                0.3655,
            ]
        )

    x_sol = r_shock * lambda_shock
    rho_sol = rho_shock * g_shock
    u_sol = u_shock * f_shock
    p_sol = p_shock * h_shock

    x_sol = np.append(x_sol, np.linspace(x_sol[-1] + 0.0001, DOMAIN[1], 500))
    rho_sol = np.append(rho_sol, np.full(500, RHO_0))
    u_sol = np.append(u_sol, np.full(500, U_0))
    p_sol = np.append(p_sol, np.full(500, P_0))

    return x_sol, rho_sol, u_sol, p_sol


if __name__ == "__main__":
    main()
