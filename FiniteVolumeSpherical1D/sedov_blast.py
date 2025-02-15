import numpy as np

from .system import System


def get_initial_system(num_cells: int) -> System:
    gamma = 5.0 / 3.0
    system = System(num_cells=num_cells, gamma=gamma, boundary_condition="transmissive")
    system.density.fill(1.0)
    system.velocity.fill(0.0)
    system.pressure.fill(1e-5)
    for i in range(num_cells + system.num_ghosts_cells):
        system.mid_points[i] = (i + 0.5) / num_cells
        system.volume[i] = (
            4.0 / 3.0 * np.pi * (((i + 1) / num_cells) ** 3 - (i / num_cells) ** 3)
        )
        system.surface_area[i] = 4.0 * np.pi * ((i / num_cells) ** 2)
    system.pressure[0] = (3.0 * (gamma - 1.0) * 1.0) / (
        (3.0 + 1.0) * np.pi * (3.0 / num_cells) ** 3
    )
    system.pressure[1] = (3.0 * (gamma - 1.0) * 1.0) / (
        (3.0 + 1.0) * np.pi * (3.0 / num_cells) ** 3
    )
    system.pressure[2] = (3.0 * (gamma - 1.0) * 1.0) / (
        (3.0 + 1.0) * np.pi * (3.0 / num_cells) ** 3
    )

    system.convert_primitive_to_conserved()
    system.set_boundary_condition()

    return system


def get_reference_sol(
    gamma: float, tf: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_ref = np.arange(0.0, 1.0, 0.001)
    # rho_ref = sol[:, 0]
    # u_ref = sol[:, 1]
    # p_ref = sol[:, 2]
    rho_ref = np.zeros_like(x_ref)
    u_ref = np.zeros_like(x_ref)
    p_ref = np.zeros_like(x_ref)

    return x_ref, rho_ref, u_ref, p_ref
