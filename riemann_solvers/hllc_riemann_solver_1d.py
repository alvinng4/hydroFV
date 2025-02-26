"""
HLLC Riemann solver for the 1D Euler equations.

Usage:
    solver = HLLCRiemannSolver1D()

References:
    1. Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics,
       3rd ed. Springer., 2009.

Author: Ching-Yin Ng
Date: 2025-2-26
"""

import math

# import warnings
from typing import Tuple

import numpy as np

from . import riemann_solver
from . import utils


class HLLCRiemannSolver1D:
    @staticmethod
    def solve_system_flux(
        gamma: float,
        rho: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        tol: float = 1e-6,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Solve the Riemann problem for the entire system.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho : np.ndarray
            Density of the system.
        u : np.ndarray
            Velocity of the system.
        p : np.ndarray
            Pressure of the system.
        tol : float, optional
            Tolerance for the estimation of p_star, by default 1e-6.

        Note
        ----
        The energy array is not energy per mass, but the total energy per cell.

        Returns
        -------
        flux_mass : np.ndarray
            Mass flux.
        flux_momentum : np.ndarray
            Momentum flux.
        flux_energy : np.ndarray
            Energy flux.
        """
        flux_mass = np.zeros(len(rho) - 1)
        flux_momentum = np.zeros(len(rho) - 1)
        flux_energy = np.zeros(len(rho) - 1)

        for i in range(len(rho) - 1):
            rho_L = rho[i]
            u_L = u[i]
            p_L = p[i]

            rho_R = rho[i + 1]
            u_R = u[i + 1]
            p_R = p[i + 1]

            flux_mass[i], flux_momentum[i], flux_energy[i] = (
                HLLCRiemannSolver1D.solve_flux(
                    gamma,
                    rho_L,
                    u_L,
                    p_L,
                    rho_R,
                    u_R,
                    p_R,
                    tol,
                )
            )

        return flux_mass, flux_momentum, flux_energy

    @staticmethod
    def solve_flux(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        tol: float = 1e-6,
    ) -> Tuple[float, float, float]:
        """Solve the Riemann problem.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_L : float
            Density of the left state.
        u_L : float
            Velocity of the left state.
        p_L : float
            Pressure of the left state.
        rho_R : float
            Density of the right state.
        u_R : float
            Velocity of the right state.
        p_R : float
            Pressure of the right state.
        tol : float, optional
            Tolerance for the estimation of p_star, by default 1e-6.

        Note
        ----
        The energy is not energy per mass, but the total energy per cell.

        Returns
        -------
        flux_mass : float
            Mass flux.
        flux_momentum : float
            Momentum flux.
        flux_energy : float
            Energy flux.
        """
        if gamma <= 1.0:
            raise ValueError("The adiabatic index needs to be larger than 1!")

        ### Compute the sound speeds ###
        if rho_L > 0.0:
            a_L = float(utils.get_sound_speed(gamma, rho_L, p_L))
        else:
            # Vacuum
            # Assuming isentropic-type EOS, p = p(rho), see Toro [1]
            a_L = 0.0

        if rho_R > 0.0:
            a_R = float(utils.get_sound_speed(gamma, rho_R, p_R))
        else:
            # Vacuum
            # Assuming isentropic-type EOS, p = p(rho), see Toro [1]
            a_R = 0.0

        ### Check for vacuum or vacuum generation ###
        # (1) If left or right state is vacuum, or
        # (2) pressure positivity condition is met
        if (rho_L <= 0.0 or rho_R <= 0.0) or (
            ((2.0 / (gamma + 1.0)) * (a_L + a_R)) <= (u_R - u_L)
        ):
            rho, u, p = riemann_solver.solve_vacuum(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R
            )
            flux_mass = rho * u
            rho_u_u = flux_mass * u
            flux_momentum = rho_u_u + p
            flux_energy = (p * (gamma / (gamma - 1.0)) + 0.5 * rho_u_u) * u
            return flux_mass, flux_momentum, flux_energy

        ### Estimate the wave speeds ###
        p_star = riemann_solver.guess_p(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )
        q_L = HLLCRiemannSolver1D.compute_q_L_or_R(gamma, p_L, p_star)
        q_R = HLLCRiemannSolver1D.compute_q_L_or_R(gamma, p_R, p_star)

        S_L = u_L - a_L * q_L
        S_R = u_R + a_R * q_R

        S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / (
            rho_L * (S_L - u_L) - rho_R * (S_R - u_R)
        )

        ### Compute the fluxes ###
        if 0.0 <= S_L:
            energy_density_L = (p_L / rho_L) / (gamma - 1.0) + 0.5 * u_L * u_L
            flux_mass_L = rho_L * u_L
            flux_momentum_L = rho_L * u_L * u_L + p_L
            flux_energy_L = (rho_L * energy_density_L + p_L) * u_L
            return flux_mass_L, flux_momentum_L, flux_energy_L
        elif S_L <= 0.0 <= S_star:
            energy_density_L = (p_L / rho_L) / (gamma - 1.0) + 0.5 * u_L * u_L
            flux_mass_L = rho_L * u_L
            flux_momentum_L = rho_L * u_L * u_L + p_L
            flux_energy_L = (rho_L * energy_density_L + p_L) * u_L

            rho_star_L = rho_L * (S_L - u_L) / (S_L - S_star)
            momentum_star_L = rho_star_L * S_star
            energy_star_L = rho_star_L * (
                energy_density_L
                + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L)))
            )

            flux_mass_star_L = flux_mass_L + S_L * (rho_star_L - rho_L)
            flux_momentum_star_L = flux_momentum_L + S_L * (
                momentum_star_L - rho_L * u_L
            )
            flux_energy_star_L = flux_energy_L + S_L * (
                energy_star_L - rho_L * energy_density_L
            )

            return flux_mass_star_L, flux_momentum_star_L, flux_energy_star_L
        elif S_star <= 0.0 <= S_R:
            energy_density_R = (p_R / rho_R) / (gamma - 1.0) + 0.5 * u_R * u_R
            flux_mass_R = rho_R * u_R
            flux_momentum_R = rho_R * u_R * u_R + p_R
            flux_energy_R = (rho_R * energy_density_R + p_R) * u_R

            rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star)
            momentum_star_R = rho_star_R * S_star
            energy_star_R = rho_star_R * (
                energy_density_R
                + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R)))
            )

            flux_mass_star_R = flux_mass_R + S_R * (rho_star_R - rho_R)
            flux_momentum_star_R = flux_momentum_R + S_R * (
                momentum_star_R - rho_R * u_R
            )
            flux_energy_star_R = flux_energy_R + S_R * (
                energy_star_R - rho_R * energy_density_R
            )

            return flux_mass_star_R, flux_momentum_star_R, flux_energy_star_R
        elif S_R <= 0.0:
            energy_density_R = (p_R / rho_R) / (gamma - 1.0) + 0.5 * u_R * u_R
            flux_mass_R = rho_R * u_R
            flux_momentum_R = rho_R * u_R * u_R + p_R
            flux_energy_R = (rho_R * energy_density_R + p_R) * u_R
            return flux_mass_R, flux_momentum_R, flux_energy_R
        else:
            raise ValueError("Invalid wave speeds!")

    @staticmethod
    def compute_q_L_or_R(gamma, p_L_or_R, p_star):
        """Compute the q_L or q_R function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        p_L_or_R : float
            Pressure of the left or right state.
        p_star : float
            Pressure in the star region.

        Returns
        -------
        float
            Value of the q_L or q_R function.
        """
        if p_star <= p_L_or_R:
            return 1.0
        else:
            return math.sqrt(
                1.0 + (0.5 * (1.0 + 1.0 / gamma)) * (p_star / p_L_or_R - 1.0)
            )
