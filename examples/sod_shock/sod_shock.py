"""
Sod shock tube test

Usage:
    python sod_shock.py

Author: Ching-Yin Ng
Date: 2025-2-22
"""

import sys
import timeit
import warnings
from pathlib import Path

sys.path.append("../../")

import matplotlib.pyplot as plt
import numpy as np
import rich.progress

import FiniteVolume1D
import riemann_solvers

RIEMANN_SOLVER = "hllc"
COORD_SYS = "spherical_1d"  # "cartesian_1d", "cylindrical_1d", or "spherical_1d"
NUM_TOTAL_CELLS = 512
NUM_GHOST_CELLS_SIDE = 1
NUM_CELLS = NUM_TOTAL_CELLS - 2 * NUM_GHOST_CELLS_SIDE
SOLVER = "godunov_first_order"  # "godunov_first_order" or "random_choice"

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


def main() -> None:
    global RIEMANN_SOLVER
    assert COORD_SYS in ["cartesian_1d", "cylindrical_1d", "spherical_1d"]

    cfl = 0.9
    tf = 0.25

    system = get_initial_system(NUM_CELLS, COORD_SYS)

    if SOLVER == "random_choice":
        rng = FiniteVolume1D.random_choice.VanDerCorputSequenceGenerator()

        if RIEMANN_SOLVER != "exact":
            msg = "Only exact Riemann solver is supported for random_choice solver. Switching to exact Riemann solver."
            warnings.warn(msg)
            RIEMANN_SOLVER = "exact"

        if cfl > 0.5:
            msg = "The Courant number should be less than 0.5 for random_choice solver. Switching to 0.4."
            warnings.warn(msg)
            cfl = 0.4


    t = 0.0
    num_steps = 0
    start = timeit.default_timer()
    with rich.progress.Progress() as progress:
        print("Simulation in progress...")
        task = progress.add_task("", total=tf)
        while t < tf:
            if num_steps <= 50:
                dt = FiniteVolume1D.utils.get_time_step(cfl * 0.2, system)
            else:
                dt = FiniteVolume1D.utils.get_time_step(cfl, system)

            if t + dt > tf:
                dt = tf - t

            if SOLVER == "godunov_first_order":
                FiniteVolume1D.godunov_first_order.solving_step(
                    system, dt, RIEMANN_SOLVER
                )
            elif SOLVER == "random_choice":
                FiniteVolume1D.random_choice.solving_step(
                    system, dt, RIEMANN_SOLVER, rng
                )

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    end = timeit.default_timer()
    print(f"Done! Num steps: {num_steps}, Time: {end - start:.3f}s")
    print()

    # Plot the reference solution and the actual solution
    x_sol, rho_sol, u_sol, p_sol = get_reference_sol(tf)

    _, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(x_sol, rho_sol, "r-")
    axs[0].plot(
        system.mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        system.density[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(x_sol, u_sol, "r-")
    axs[1].plot(
        system.mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        system.velocity[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(x_sol, p_sol, "r-")
    axs[2].plot(
        system.mid_points[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        system.pressure[NUM_GHOST_CELLS_SIDE:-NUM_GHOST_CELLS_SIDE],
        "k.",
        markersize=2,
    )
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


def get_initial_system(num_cells: int, coord_sys: str) -> FiniteVolume1D.system.System:
    system = FiniteVolume1D.system.System(
        num_cells=num_cells,
        gamma=GAMMA,
        coord_sys=coord_sys,
        domain=(0.0, 1.0),
        num_ghost_cells_side=NUM_GHOST_CELLS_SIDE,
        left_boundary_condition=LEFT_BOUNDARY_CONDITION,
        right_boundary_condition=RIGHT_BOUNDARY_CONDITION,
    )
    for i in range(system.total_num_cells):
        if (i + 0.5) / system.total_num_cells < DISCONTINUITY_POS:
            system.density[i] = RHO_L
            system.velocity[i] = U_L
            system.pressure[i] = P_L
        else:
            system.density[i] = RHO_R
            system.velocity[i] = U_R
            system.pressure[i] = P_R

    system.set_boundary_condition()
    system.convert_primitive_to_conserved()

    return system


def get_reference_sol(
    tf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the reference solution of the Sod shock tube problem.

    Parameters
    ----------
    tf : float
        Final time.

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
        return _get_cartesian_1d_reference_sol(tf)
    else:
        sol_path = Path(__file__).parent / f"sol_{COORD_SYS}.npz"
        if sol_path.exists():
            data = np.load(sol_path)
            return (
                data["mid_points"],
                data["density"],
                data["velocity"],
                data["pressure"],
            )
        else:
            return _get_noncartesian_1d_reference_sol(tf, sol_path)


def _get_cartesian_1d_reference_sol(
    tf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the reference solution of the Sod shock tube problem for cartesian_1d.

    Parameters
    ----------
    tf : float
        Final time.

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
    x_sol = np.arange(0.0, 1.0, 0.001)
    sol = np.array(
        [
            riemann_solvers.solve(
                gamma=GAMMA,
                rho_L=RHO_L,
                u_L=U_L,
                p_L=P_L,
                rho_R=RHO_R,
                u_R=U_R,
                p_R=P_R,
                speed=(x - DISCONTINUITY_POS) / tf,
                dim=1,
                solver="exact",
            )
            for x in x_sol
        ]
    )
    rho_sol = sol[:, 0]
    u_sol = sol[:, 1]
    p_sol = sol[:, 2]

    return x_sol, rho_sol, u_sol, p_sol


def _get_noncartesian_1d_reference_sol(
    tf: float, save_path: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get the reference solution of the Sod shock tube problem for cylindrical_1d or spherical_1d geometry.

    Parameters
    ----------
    tf : float
        Final time.

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
    cfl = 0.1
    system = get_initial_system(2560, COORD_SYS)

    rng = FiniteVolume1D.random_choice.VanDerCorputSequenceGenerator()

    t = 0.0
    num_steps = 0
    start = timeit.default_timer()
    with rich.progress.Progress() as progress:
        print("Obtaining reference solution (with RCM)...")
        task = progress.add_task("", total=tf)
        while t < tf:
            if num_steps <= 50:
                dt = FiniteVolume1D.utils.get_time_step(cfl * 0.2, system)
            else:
                dt = FiniteVolume1D.utils.get_time_step(cfl, system)

            if t + dt > tf:
                dt = tf - t

            FiniteVolume1D.random_choice.solving_step(system, dt, "exact", rng)

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    end = timeit.default_timer()
    print(f"Done! Num steps: {num_steps}, Time: {end - start:.3f}s")

    # Save the solution
    np.savez(
        save_path,
        mid_points=system.mid_points,
        density=system.density,
        velocity=system.velocity,
        pressure=system.pressure,
    )
    print(f'Reference solution saved to "{save_path}".')

    return system.mid_points, system.density, system.velocity, system.pressure


if __name__ == "__main__":
    main()
