import sys
import timeit

sys.path.append("../")

import matplotlib.pyplot as plt
import rich.progress

import FiniteVolume1D
import FiniteVolumeSpherical1D

RIEMANN_SOLVER = "hllc"
COORD_SYS = "cartesian_1d"
NUM_CELLS = 128

def main() -> None:
    assert COORD_SYS in ["cartesian_1d", "spherical_1d"]

    cfl = 0.9
    tf = 0.2

    if COORD_SYS == "cartesian_1d":
        simulate_cartesian_1d(NUM_CELLS, cfl, tf)
    elif COORD_SYS == "spherical_1d":
        simulate_spherical_1d(NUM_CELLS, cfl, tf)

def simulate_cartesian_1d(num_cells: int, cfl: float, tf: float) -> None:
    system = FiniteVolume1D.sod_shock.get_initial_system(num_cells)

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
    x_ref, rho_ref, u_ref, p_ref = FiniteVolume1D.sod_shock.get_reference_sol(
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

def simulate_spherical_1d(num_cells: int, cfl: float, tf: float) -> None:
    system = FiniteVolumeSpherical1D.sod_shock.get_initial_system(num_cells)

    t = 0.0
    num_steps = 0
    start = timeit.default_timer()
    with rich.progress.Progress() as progress:
        print("Simulation in progress...")
        task = progress.add_task("", total=tf)
        while t < tf:
            if num_steps <= 50:
                dt = FiniteVolumeSpherical1D.utils.get_time_step(cfl * 0.2, system)
            else:
                dt = FiniteVolumeSpherical1D.utils.get_time_step(cfl, system)

            if t + dt > tf:
                dt = tf - t

            FiniteVolumeSpherical1D.godunov_first_order.solving_step(system, dt, RIEMANN_SOLVER)

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    end = timeit.default_timer()
    print(f"Done! Num steps: {num_steps}, Time: {end - start:.3f}s")

    # plot the reference solution and the actual solution
    x_ref, rho_ref, u_ref, p_ref = FiniteVolumeSpherical1D.sod_shock.get_reference_sol(
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


if __name__ == "__main__":
    main()
