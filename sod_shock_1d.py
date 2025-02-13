import matplotlib.pyplot as plt
import rich.progress

from FiniteVolume1D.exact_riemann_solver import ExactRiemannSolver
from FiniteVolume1D import godunov_first_order
from FiniteVolume1D import utils
from FiniteVolume1D import sod_shock


def main() -> None:
    num_cell = 100  # Note the CFL condition, delta x cannot be too small
    gamma = 5.0 / 3.0

    cfl = 0.9
    tf = 0.2

    riemann_solver = ExactRiemannSolver()

    cells = sod_shock.cells_initial_condition(num_cell, gamma)

    t = 0.0
    num_steps = 0

    with rich.progress.Progress() as progress:
        print("Simulation in progress...")
        task = progress.add_task("", total=tf)
        while t < tf:
            if num_steps <= 50:
                dt = utils.get_time_step(cfl * 0.2, cells, gamma)
            else:
                dt = utils.get_time_step(cfl, cells, gamma)

            if t + dt > tf:
                dt = tf - t
            
            godunov_first_order.solving_step(cells, dt, gamma, riemann_solver)

            t += dt
            num_steps += 1
            progress.update(task, completed=t)

    print("Done!")
    print(f"Number of steps: {num_steps}")

    # plot the reference solution and the actual solution
    x_ref, rho_ref, u_ref, p_ref = sod_shock.get_reference_sol(gamma, tf, riemann_solver)

    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    axs[0].plot(
        [cell._midpoint for cell in cells], [cell._density for cell in cells], "k."
    )
    axs[0].plot(x_ref, rho_ref, "r-")
    axs[0].set_xlabel("Position")
    axs[0].set_ylabel("Density")

    axs[1].plot(
        [cell._midpoint for cell in cells], [cell._velocity for cell in cells], "k."
    )
    axs[1].plot(x_ref, u_ref, "r-")
    axs[1].set_xlabel("Position")
    axs[1].set_ylabel("Velocity")

    axs[2].plot(
        [cell._midpoint for cell in cells], [cell._pressure for cell in cells], "k."
    )
    axs[2].plot(x_ref, p_ref, "r-")
    axs[2].set_xlabel("Position")
    axs[2].set_ylabel("Pressure")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
