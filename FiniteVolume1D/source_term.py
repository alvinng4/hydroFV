from typing import Tuple

import numpy as np

from .system import System


def add_spherical_source_term(
    system: System,
    dt: float,
) -> None:
    """Add the spherical source term for the spherical 1D euler equations."""
    d_mass, d_momentum, d_energy = rk2(system, dt)

    system.mass[1:-1] += d_mass
    system.momentum[1:-1] += d_momentum
    system.energy[1:-1] += d_energy

    system.convert_conserved_to_primitive()
    system.set_boundary_condition()


def compute_source_term(
    gamma: float,
    pre_factor: np.ndarray,
    density: np.ndarray,
    specific_momentum: np.ndarray,
    specific_energy: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pressure = (gamma - 1) * (specific_energy - 0.5 * specific_momentum**2 / density)
    d_mass = pre_factor * specific_momentum
    d_momentum = pre_factor * specific_momentum**2 / density
    d_energy = pre_factor * specific_momentum * (specific_energy + pressure) / density

    return d_mass, d_momentum, d_energy


def euler(system: System, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_factor = -2.0 / system.mid_points[1:-1] * system.volume[1:-1]
    d_mass, d_momentum, d_energy = compute_source_term(
        system.gamma,
        pre_factor * dt,
        system.density[1:-1],
        system.momentum[1:-1] / system.volume[1:-1],
        system.energy[1:-1] / system.volume[1:-1],
    )

    return d_mass, d_momentum, d_energy


def rk2(system: System, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_factor = -2.0 / system.mid_points[1:-1] * system.volume[1:-1]
    k1_mass, k1_momentum, k1_energy = compute_source_term(
        system.gamma,
        pre_factor,
        system.density[1:-1],
        system.momentum[1:-1] / system.volume[1:-1],
        system.energy[1:-1] / system.volume[1:-1],
    )

    temp_mass = system.mass[1:-1] + k1_mass * dt
    temp_momentum = system.momentum[1:-1] + k1_momentum * dt
    temp_energy = system.energy[1:-1] + k1_energy * dt

    k2_mass, k2_momentum, k2_energy = compute_source_term(
        system.gamma,
        pre_factor,
        temp_mass / system.volume[1:-1],
        temp_momentum / system.volume[1:-1],
        temp_energy / system.volume[1:-1],
    )

    d_mass = 0.5 * (k1_mass + k2_mass) * dt
    d_momentum = 0.5 * (k1_momentum + k2_momentum) * dt
    d_energy = 0.5 * (k1_energy + k2_energy) * dt

    return d_mass, d_momentum, d_energy


def rk4(system: System, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pre_factor = -2.0 / system.mid_points[1:-1] * system.volume[1:-1]
    k1_mass, k1_momentum, k1_energy = compute_source_term(
        system.gamma,
        pre_factor,
        system.density[1:-1],
        system.momentum[1:-1] / system.volume[1:-1],
        system.energy[1:-1] / system.volume[1:-1],
    )

    temp_mass = system.mass[1:-1] + k1_mass * (0.5 * dt)
    temp_momentum = system.momentum[1:-1] + k1_momentum * (0.5 * dt)
    temp_energy = system.energy[1:-1] + k1_energy * (0.5 * dt)

    k2_mass, k2_momentum, k2_energy = compute_source_term(
        system.gamma,
        pre_factor,
        temp_mass / system.volume[1:-1],
        temp_momentum / system.volume[1:-1],
        temp_energy / system.volume[1:-1],
    )

    temp_mass = system.mass[1:-1] + k2_mass * (0.5 * dt)
    temp_momentum = system.momentum[1:-1] + k2_momentum * (0.5 * dt)
    temp_energy = system.energy[1:-1] + k2_energy * (0.5 * dt)

    k3_mass, k3_momentum, k3_energy = compute_source_term(
        system.gamma,
        pre_factor,
        temp_mass / system.volume[1:-1],
        temp_momentum / system.volume[1:-1],
        temp_energy / system.volume[1:-1],
    )

    temp_mass = system.mass[1:-1] + k3_mass * dt
    temp_momentum = system.momentum[1:-1] + k3_momentum * dt
    temp_energy = system.energy[1:-1] + k3_energy * dt

    k4_mass, k4_momentum, k4_energy = compute_source_term(
        system.gamma,
        pre_factor,
        temp_mass / system.volume[1:-1],
        temp_momentum / system.volume[1:-1],
        temp_energy / system.volume[1:-1],
    )

    d_mass = (k1_mass + 2 * k2_mass + 2 * k3_mass + k4_mass) * dt / 6
    d_momentum = (
        (k1_momentum + 2 * k2_momentum + 2 * k3_momentum + k4_momentum) * dt / 6
    )
    d_energy = (k1_energy + 2 * k2_energy + 2 * k3_energy + k4_energy) * dt / 6

    return d_mass, d_momentum, d_energy
