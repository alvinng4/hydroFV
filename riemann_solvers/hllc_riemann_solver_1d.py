"""
HLLC Riemann solver for the 1D Euler equations.

Usage:
    solver = HLLCRiemannSolverCartesian1D()

References:
    1. Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics,
       3rd ed. Springer., 2009.
"""

import math

# import warnings
from typing import Tuple

import numpy as np

from . import utils


class HLLCRiemannSolverCartesian1D:
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
                HLLCRiemannSolverCartesian1D.solve_flux(
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

        ### Handle vacuum ###
        if rho_L <= 0.0 and rho_R <= 0.0:
            return 0.0, 0.0, 0.0
        elif rho_L <= 0.0 or rho_R <= 0.0:
            raise NotImplementedError

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

        ### Estimate the wave speeds ###
        p_star = HLLCRiemannSolverCartesian1D.guess_p(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )
        q_L = HLLCRiemannSolverCartesian1D.compute_q_L_or_R(gamma, p_L, p_star)
        q_R = HLLCRiemannSolverCartesian1D.compute_q_L_or_R(gamma, p_R, p_star)

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
                energy_density_L + (S_star - u_L) * (S_star + p_L / (rho_L * (S_L - u_L)))
            )

            flux_mass_star_L = flux_mass_L + S_L * (rho_star_L - rho_L)
            flux_momentum_star_L = flux_momentum_L + S_L * (
                momentum_star_L - rho_L * u_L
            )
            flux_energy_star_L = flux_energy_L + S_L * (energy_star_L - rho_L * energy_density_L)

            return flux_mass_star_L, flux_momentum_star_L, flux_energy_star_L
        elif S_star <= 0.0 <= S_R:
            energy_density_R = (p_R / rho_R) / (gamma - 1.0) + 0.5 * u_R * u_R
            flux_mass_R = rho_R * u_R
            flux_momentum_R = rho_R * u_R * u_R + p_R
            flux_energy_R = (rho_R * energy_density_R + p_R) * u_R

            rho_star_R = rho_R * (S_R - u_R) / (S_R - S_star)
            momentum_star_R = rho_star_R * S_star
            energy_star_R = rho_star_R * (
                energy_density_R + (S_star - u_R) * (S_star + p_R / (rho_R * (S_R - u_R)))
            )

            flux_mass_star_R = flux_mass_R + S_R * (rho_star_R - rho_R)
            flux_momentum_star_R = flux_momentum_R + S_R * (
                momentum_star_R - rho_R * u_R
            )
            flux_energy_star_R = flux_energy_R + S_R * (energy_star_R - rho_R * energy_density_R)

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

    @staticmethod
    def riemann_A_L_or_R(gamma: float, rho_X: float) -> float:
        """Riemann A_L or A_R function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_X : float
            Density of the left or right state.

        Returns
        -------
        float
            Value of the A_L or A_R function.
        """
        return 2.0 / ((gamma + 1.0) * rho_X)

    @staticmethod
    def riemann_B_L_or_R(gamma: float, p_X: float) -> float:
        """Riemann B_L or B_R function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        p_X : float
            Pressure of the left or right state.

        Returns
        -------
        float
            Value of the B_L or B_R function.
        """
        return ((gamma - 1.0) / (gamma + 1.0)) * p_X

    @staticmethod
    def guess_p(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        tol: float,
    ) -> float:
        """Get an initial guess for the pressure in the middle state.

        Parameters
        ----------
        rho_L : float
            Density of the left state.
        u_L : float
            Velocity of the left state.
        p_L : float
            Pressure of the left state.
        a_L : float
            Sound speed of the left state.
        rho_R : float
            Density of the right state.
        u_R : float
            Velocity of the right state.
        p_R : float
            Pressure of the right state.
        a_R : float
            Sound speed of the right state.
        tol : float
            Tolerance for the initial guess.

        Returns
        -------
        p_guess : float
           Initial guess for the pressure in the middle state.
        """
        p_min = min(p_L, p_R)
        p_max = max(p_L, p_R)

        ppv = 0.5 * (p_L + p_R) - 0.125 * (u_R - u_L) * (rho_L + rho_R) * (a_L + a_R)

        # Select PVRS Riemann solver
        if (p_max / p_min) <= 2.0 and p_min <= ppv and ppv <= p_max:
            p_guess = ppv

        # Select Two-Rarefaction Riemann solver
        elif ppv < p_min:
            gamma_minus_one = gamma - 1.0
            gamma_minus_one_over_two_gamma = gamma_minus_one / (2.0 * gamma)
            p_guess = (
                (a_L + a_R - 0.5 * gamma_minus_one * (u_R - u_L))
                / (
                    a_L / (p_L**gamma_minus_one_over_two_gamma)
                    + a_R / (p_R**gamma_minus_one_over_two_gamma)
                )
            ) ** (2.0 * gamma / gamma_minus_one)

        # Select Two-Shock Riemann solver with PVRS as estimate
        else:
            A_L = HLLCRiemannSolverCartesian1D.riemann_A_L_or_R(gamma, rho_L)
            B_L = HLLCRiemannSolverCartesian1D.riemann_B_L_or_R(gamma, p_L)
            g_L = math.sqrt(A_L / (ppv + B_L))

            A_R = HLLCRiemannSolverCartesian1D.riemann_A_L_or_R(gamma, rho_R)
            B_R = HLLCRiemannSolverCartesian1D.riemann_B_L_or_R(gamma, p_R)
            g_R = math.sqrt(A_R / (ppv + B_R))

            p_guess = (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R)

        # Prevent negative value
        return max(tol, p_guess)
