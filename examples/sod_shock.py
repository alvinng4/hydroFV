import sys
import timeit

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import rich.progress

import FiniteVolume1D
import riemann_solvers

RIEMANN_SOLVER = "hllc"
COORD_SYS = "cartesian_1d"
NUM_CELLS = 128


def main() -> None:
    assert COORD_SYS in ["cartesian_1d", "spherical_1d"]

    cfl = 0.9
    tf = 0.2

    system = get_initial_system(NUM_CELLS, COORD_SYS)

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

            FiniteVolume1D.godunov_first_order.solving_step(system, dt, RIEMANN_SOLVER)

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    end = timeit.default_timer()
    print(f"Done! Num steps: {num_steps}, Time: {end - start:.3f}s")

    # plot the reference solution and the actual solution
    x_ref, rho_ref, u_ref, p_ref = get_reference_sol(
        system.gamma,
        tf,
    )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(system.mid_points[1:-1], system.density[1:-1], "k.")
    axs[0].plot(x_ref, rho_ref, "r-")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(system.mid_points[1:-1], system.velocity[1:-1], "k.")
    axs[1].plot(x_ref, u_ref, "r-")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(system.mid_points[1:-1], system.pressure[1:-1], "k.")
    axs[2].plot(x_ref, p_ref, "r-")
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


def get_initial_system(num_cells: int, coord_sys: str) -> FiniteVolume1D.system.System:
    system = FiniteVolume1D.system.System(
        num_cells=num_cells,
        gamma=5.0 / 3.0,
        left_boundary_condition="reflective",
        right_boundary_condition="reflective",
    )
    total_num_cells = num_cells + system.num_ghosts_cells
    system.velocity.fill(0.0)
    if coord_sys == "cartesian_1d":
        system.volume.fill(1.0 / total_num_cells)
        system.surface_area.fill(1.0)
        for i in range(total_num_cells):
            system.mid_points[i] = (i + 0.5) / total_num_cells

            if i < num_cells / 2:
                system.density[i] = 1.0
                system.pressure[i] = 1.0
            else:
                system.density[i] = 0.125
                system.pressure[i] = 0.1
    elif coord_sys == "spherical_1d":
        for i in range(total_num_cells):
            system.mid_points[i] = (i + 0.5) / total_num_cells
            system.volume[i] = (
                4.0
                / 3.0
                * np.pi
                * (((i + 1) / total_num_cells) ** 3 - (i / total_num_cells) ** 3)
            )
            system.surface_area[i] = 4.0 * np.pi * ((i / total_num_cells) ** 2)

            if i < num_cells / 2:
                system.density[i] = 1.0
                system.pressure[i] = 1.0
            else:
                system.density[i] = 0.125
                system.pressure[i] = 0.1
    else:
        raise ValueError("Invalid coord_sys")

    system.convert_primitive_to_conserved()
    system.set_boundary_condition()

    return system


def get_reference_sol(
    gamma: float, tf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.arange(0.0, 1.0, 0.001)
    sol = np.array(
        [
            riemann_solvers.solve(
                gamma=gamma,
                rho_L=1.0,
                u_L=0.0,
                p_L=1.0,
                rho_R=0.125,
                u_R=0.0,
                p_R=0.1,
                speed=(x - 0.5) / tf,
                coord_sys="cartesian_1d",
                solver="exact",
            )
            for x in x_ref
        ]
    )
    rho_ref = sol[:, 0]
    u_ref = sol[:, 1]
    p_ref = sol[:, 2]

    return x_ref, rho_ref, u_ref, p_ref


if __name__ == "__main__":
    main()
