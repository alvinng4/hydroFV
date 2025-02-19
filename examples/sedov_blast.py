"""

References
----------
High-order schemes for cylindrical/spherical geometries with cylindrical/spherical symmetry, Sheng Wang and Eric Johnsen
"""

import sys
import timeit

sys.path.append("../")

import matplotlib.pyplot as plt
import numpy as np
import rich.progress

import FiniteVolume1D
# import riemann_solvers

RIEMANN_SOLVER = "hllc"
COORD_SYS = "spherical_1d"
NUM_CELLS = 256


def main() -> None:
    assert COORD_SYS in ["cartesian_1d", "spherical_1d"]

    cfl = 0.9
    tf = 2.0

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
    # x_ref, rho_ref, u_ref, p_ref = get_reference_sol(
    #     system.gamma,
    #     tf,
    # )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(system.mid_points[1:-1], system.density[1:-1], "k.")
    # axs[0].plot(x_ref, rho_ref, "r-")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(system.mid_points[1:-1], system.velocity[1:-1], "k.")
    # axs[1].plot(x_ref, u_ref, "r-")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(system.mid_points[1:-1], system.pressure[1:-1], "k.")
    # axs[2].plot(x_ref, p_ref, "r-")
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


def get_initial_system(num_cells: int, coord_sys: str) -> FiniteVolume1D.system.System:
    system = FiniteVolume1D.system.System(
        num_cells=num_cells,
        gamma=5.0 / 3.0,
        coord_sys=coord_sys,
        left_boundary_condition="reflective",
        right_boundary_condition="transmissive",
    )
    system.density.fill(1.0)
    system.velocity.fill(0.0)
    system.pressure.fill(1e-5)
    for i in range(system.total_num_cells):
        system.cell_left[i] = i / system.total_num_cells
        system.cell_right[i] = (i + 1) / system.total_num_cells

    if coord_sys == "cartesian_1d":
        alpha = 0.0
    elif coord_sys == "spherical_1d":
        alpha = 2.0
    else:
        raise ValueError("Invalid coord_sys")

    system.pressure[1] = (3.0 * (system.gamma - 1.0) * 1.0) / (
        (alpha + 1.0) * np.pi * (3.0 / system.total_num_cells) ** alpha
    )
    system.pressure[2] = (3.0 * (system.gamma - 1.0) * 1.0) / (
        (alpha + 1.0) * np.pi * (3.0 / system.total_num_cells) ** alpha
    )
    system.pressure[3] = (3.0 * (system.gamma - 1.0) * 1.0) / (
        (alpha + 1.0) * np.pi * (3.0 / system.total_num_cells) ** alpha
    )

    system.compute_volume()
    system.compute_surface_area()
    system.compute_mid_points()
    system.convert_primitive_to_conserved()
    system.set_boundary_condition()

    return system


# def get_reference_sol(
#     gamma: float, tf: float,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
#     gamma = 5. / 3.
#     R =
#     x_ref = np.arange(0.0, 1.0, 0.001)
#     sol = np.array(
#         [
#             riemann_solvers.solve(
#                 gamma=gamma,
#                 rho_L=1.0,
#                 u_L=0.0,
#                 p_L=1.0,
#                 rho_R=0.125,
#                 u_R=0.0,
#                 p_R=0.1,
#                 speed=(x - 0.5) / tf,
#                 coord_sys="cartesian_1d",
#                 solver="exact",
#             )
#             for x in x_ref
#         ]
#     )
#     rho_ref = sol[:, 0]
#     u_ref = sol[:, 1]
#     p_ref = sol[:, 2]

#     return x_ref, rho_ref, u_ref, p_ref

if __name__ == "__main__":
    main()
