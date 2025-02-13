import sys
sys.path.append("../")

import matplotlib.pyplot as plt
import rich.progress

from FiniteVolumeSpherical.exact_riemann_solver import ExactRiemannSolver
from FiniteVolumeSpherical import godunov_first_order
from FiniteVolumeSpherical import utils
from FiniteVolumeSpherical import sod_shock


def main() -> None:
    num_cells = 100

    cfl = 0.9
    tf = 0.2

    riemann_solver = ExactRiemannSolver()
    system = sod_shock.get_initial_system(num_cells)

    t = 0.0
    num_steps = 0
    with rich.progress.Progress() as progress:
        print("Simulation in progress...")
        task = progress.add_task("", total=tf)
        while t < tf:
            if num_steps <= 50:
                dt = utils.get_time_step(cfl * 0.2, system)
            else:
                dt = utils.get_time_step(cfl, system)

            if t + dt > tf:
                dt = tf - t

            godunov_first_order.solving_step(system, dt, riemann_solver)

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    print("Done!")
    print(f"Number of steps: {num_steps}")

    # plot the reference solution and the actual solution
    x_ref, rho_ref, u_ref, p_ref = sod_shock.get_reference_sol(
        system.gamma, tf, riemann_solver
    )

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(system.mid_points[:-1], system.density[:-1], "k.")
    axs[0].plot(x_ref, rho_ref, "r-")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(system.mid_points[:-1], system.velocity[:-1], "k.")
    axs[1].plot(x_ref, u_ref, "r-")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(system.mid_points[:-1], system.pressure[:-1], "k.")
    axs[2].plot(x_ref, p_ref, "r-")
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
