"""
Exact Riemann solver for the 1D Euler equations.

Usage:
    exact_riemann_solver = ExactRiemannSolver1D()
    rho_sol, u_sol, p_sol = exact_riemann_solver.solve(
        gamma, rho_L, u_L, p_L, rho_R, u_R, p_R, speed, tol
    )
    flux_mass, flux_momentum, flux_energy = exact_riemann_solver.solve_system_flux(
        gamma, rho, u, p, speed, tol
    )

References:
    1. Toro, E. F., "The Riemann Problem for the Euler Equations" in
       Riemann Solvers and Numerical Methods for Fluid Dynamics,
       3rd ed. Springer., 2009, pp. 115-162.
    2. Press, W. H., et al., "Bracketing and Bisection" in Numerical
       Recipes in C: The Art of Scientific Computing, 2nd ed.
       Cambridge University Press, 1992, pp. 350-354.

Author: Ching-Yin Ng
Date: 2025-2-26
"""

import math
import warnings
from typing import Tuple

import numpy as np

from . import riemann_solver
from . import utils


class ExactRiemannSolver1D:
    @staticmethod
    def solve_system_flux(
        gamma: float,
        rho: np.ndarray,
        u: np.ndarray,
        p: np.ndarray,
        speed: float = 0.0,
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

        Returns
        -------
        flux_mass : np.ndarray
            Mass flux of the system.
        flux_momentum : np.ndarray
            Momentum flux of the system.
        flux_energy : np.ndarray
            Energy flux of the system
        """
        sol_rho = np.zeros(len(rho) - 1)
        sol_u = np.zeros(len(rho) - 1)
        sol_p = np.zeros(len(rho) - 1)
        for i in range(len(rho) - 1):
            rho_L = rho[i]
            u_L = u[i]
            p_L = p[i]

            rho_R = rho[i + 1]
            u_R = u[i + 1]
            p_R = p[i + 1]

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
                sol_rho[i], sol_u[i], sol_p[i] = riemann_solver.solve_vacuum(
                    gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
                )
                continue

            ### Solve for p_star and u_star ###
            p_star = ExactRiemannSolver1D.solve_p_star(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
            )
            u_star = 0.5 * (u_L + u_R) + 0.5 * (
                ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
                - ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
            )

            ### The riemann problem is solved. Now we sample the solution ###
            if speed < u_star:
                sol_rho[i], sol_u[i], sol_p[i] = ExactRiemannSolver1D.sample_left_state(
                    gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
                )
            else:
                sol_rho[i], sol_u[i], sol_p[i] = (
                    ExactRiemannSolver1D.sample_right_state(
                        gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
                    )
                )

        flux_mass = sol_rho * sol_u
        sol_rho_u_u = flux_mass * sol_u
        flux_momentum = sol_rho_u_u + sol_p
        flux_energy = (sol_p * (gamma / (gamma - 1.0)) + 0.5 * sol_rho_u_u) * sol_u

        return flux_mass, flux_momentum, flux_energy

    @staticmethod
    def solve(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        speed: float = 0.0,
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
        speed : float (optional)
            Speed S = x / t when sampling at (x, t), default is 0.0.
        tol : float (optional)
            Tolerance for the Newton-Raphson method, default is 1e-6.

        Returns
        -------
        float
            Density in the middle state.
        float
            Pressure in the middle state.
        float
            Velocity in the middle state.
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
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
            )
            return rho, u, p

        ### Solve for p_star and u_star ###
        p_star = ExactRiemannSolver1D.solve_p_star(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )
        u_star = 0.5 * (u_L + u_R) + 0.5 * (
            ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
            - ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
        )

        ### The riemann problem is solved. Now we sample the solution ###
        if speed < u_star:
            rho, u, p = ExactRiemannSolver1D.sample_left_state(
                gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
            )
        else:
            rho, u, p = ExactRiemannSolver1D.sample_right_state(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
            )

        return rho, u, p

    @staticmethod
    def solve_p_star(
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
        """Solve for the pressure in the star region, using
        the Newton-Raphson method.

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
            Tolerance for the Newton-Raphson method.

        Returns
        -------
        p_star : float
            Pressure in the star region.
        """
        p_guess = riemann_solver.guess_p(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )

        # For bisection method
        p_upper_bisection = p_guess
        f_upper_bisection = ExactRiemannSolver1D.riemann_f(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_upper_bisection
        )
        bracket_found = False

        ### Newton-Raphson method ###
        p_0 = p_guess
        max_num_iter = 100
        for _ in range(max_num_iter):
            f = ExactRiemannSolver1D.riemann_f(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_0
            )
            f_prime = ExactRiemannSolver1D.riemann_f_prime(
                gamma, rho_L, p_L, a_L, rho_R, p_R, a_R, p_0
            )
            if not bracket_found and f * f_upper_bisection < 0.0:
                bracket_found = True
                f_lower_bisection = f
                p_lower_bisection = p_0

            p = p_0 - f / f_prime

            # Failed to converge, switch to bisection method
            if p < 0.0:
                break

            if (abs(p - p_0) / abs(0.5 * (p + p_0))) < tol:
                return p

            p_0 = p

        ### Bisection method ###
        # Find lower bound and upper bound
        if not bracket_found:
            p_lower_bisection = 0.0
            f_lower_bisection = ExactRiemannSolver1D.riemann_f(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_lower_bisection
            )
            if f_lower_bisection * f_upper_bisection >= 0.0:
                p_upper_bisection *= 10.0
                num_intervals = 100
                dp = (p_upper_bisection - p_lower_bisection) / num_intervals
                for i in range(num_intervals):
                    _p_lower = p_lower_bisection + dp * i
                    _p_upper = p_lower_bisection + dp * (i + 1)

                    _f_lower = ExactRiemannSolver1D.riemann_f(
                        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, _p_lower
                    )
                    _f_upper = ExactRiemannSolver1D.riemann_f(
                        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, _p_upper
                    )

                    if _f_lower * _f_upper < 0.0:
                        p_lower_bisection = _p_lower
                        p_upper_bisection = _p_upper
                        f_lower_bisection = _f_lower
                        f_upper_bisection = _f_upper
                        break

                if _f_lower * _f_upper >= 0.0:
                    raise ValueError("Bracket not found")

        count = 0
        while True:
            p_mid = (p_upper_bisection + p_lower_bisection) / 2.0
            if (p_upper_bisection - p_lower_bisection) < tol:
                return p_mid
            f_mid = ExactRiemannSolver1D.riemann_f(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_mid
            )

            if f_lower_bisection * f_mid < 0.0:
                p_upper_bisection = p_mid
            else:
                p_lower_bisection = p_mid

            count += 1

            if count >= 500:
                warnings.warn(
                    "Bisection method failed to convert within 500 iterations"
                )

    @staticmethod
    def sample_left_state(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the Riemann problem solution for the left state.

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
        a_L : float
            Sound speed of the left state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        if p_star > p_L:
            # Shock wave
            return ExactRiemannSolver1D.sample_left_shock_wave(
                gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
            )
        else:
            # Rarefaction wave
            return ExactRiemannSolver1D.sample_left_rarefaction_wave(
                gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
            )

    @staticmethod
    def sample_right_state(
        gamma: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the Riemann problem solution for the right state.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_R : float
            Density of the right state.
        u_R : float
            Velocity of the right state.
        p_R : float
            Pressure of the right state.
        a_R : float
            Sound speed of the right state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        if p_star > p_R:
            # Shock wave
            return ExactRiemannSolver1D.sample_right_shock_wave(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
            )
        else:
            # Rarefaction wave
            return ExactRiemannSolver1D.sample_right_rarefaction_wave(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
            )

    @staticmethod
    def riemann_f_L_or_R(
        gamma: float, rho_X: float, p_X: float, a_X: float, p: float
    ) -> float:
        """Riemann f_L or f_R function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_X : float
            Density of the left or right state.
        p_X : float
            Pressure of the left or right state.
        a_X : float
            Sound speed of the left or right state.
        p : float
            Pressure of the middle state.

        Returns
        -------
        float
            Value of the f_L or f_R function.
        """
        # if p > p_X (shock)
        if p > p_X:
            A_X = riemann_solver.riemann_A_L_or_R(gamma, rho_X)
            B_X = riemann_solver.riemann_B_L_or_R(gamma, p_X)
            return (p - p_X) * math.sqrt(A_X / (p + B_X))

        # if p <= p_X (rarefaction)
        else:
            gamma_minus_one = gamma - 1.0
            return (
                2.0
                * a_X
                * ((p / p_X) ** (gamma_minus_one / (2.0 * gamma)) - 1.0)
                / gamma_minus_one
            )

    @staticmethod
    def riemann_f_L_or_R_prime(
        gamma: float, rho_X: float, p_X: float, a_X: float, p: float
    ) -> float:
        """Derivative of the Riemann f_L or f_R function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_X : float
            Density of the left or right state.
        p_X : float
            Pressure of the left or right state.
        a_X : float
            Sound speed of the left or right state.
        p : float
            Pressure of the middle state.

        Returns
        -------
        float
            Value of the derivative of the f_L or f_R function.
        """
        # if p > p_X (shock)
        if p > p_X:
            A_X = riemann_solver.riemann_A_L_or_R(gamma, rho_X)
            B_X = riemann_solver.riemann_B_L_or_R(gamma, p_X)
            return math.sqrt(A_X / (B_X + p)) * (1.0 - 0.5 * (p - p_X) / (B_X + p))

        # if p <= p_X (rarefaction)
        else:
            return ((p / p_X) ** (-0.5 * (gamma + 1.0) / gamma)) / (p_X * a_X)

    @staticmethod
    def riemann_f(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        p: float,
    ):
        """Riemann f function.

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
        p : float
            Pressure of the middle state.

        Returns
        -------
        float
            Value of the Riemann f function.
        """
        return (
            ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p)
            + ExactRiemannSolver1D.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p)
            + (u_R - u_L)
        )

    @staticmethod
    def riemann_f_prime(
        gamma: float,
        rho_L: float,
        p_L: float,
        a_L: float,
        rho_R: float,
        p_R: float,
        a_R: float,
        p: float,
    ) -> float:
        """Derivative of the Riemann f function.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_L : float
            Density of the left state.
        p_L : float
            Pressure of the left state.
        a_L : float
            Sound speed of the left state.
        rho_R : float
            Density of the right state.
        p_R : float
            Pressure of the right state.
        a_R : float
            Sound speed of the right state.
        p : float
            Pressure of the middle state.

        Returns
        -------
        float
            Value of the derivative of the Riemann f function.
        """
        return ExactRiemannSolver1D.riemann_f_L_or_R_prime(
            gamma, rho_L, p_L, a_L, p
        ) + ExactRiemannSolver1D.riemann_f_L_or_R_prime(gamma, rho_R, p_R, a_R, p)

    @staticmethod
    def sample_left_shock_wave(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the left state for a shock wave.

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
        a_L : float
            Sound speed of the left state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        p_star_over_p_L = p_star / p_L

        # Shock speed
        S_L = u_L - a_L * math.sqrt(
            (gamma + 1.0) / (2.0 * gamma) * p_star_over_p_L
            + (gamma - 1.0) / (2.0 * gamma)
        )

        # Left state regime
        if speed < S_L:
            rho = rho_L
            u = u_L
            p = p_L

        # Middle state regime
        else:
            # Gamma minus one over gamma plus one
            gmoogpo = (gamma - 1.0) / (gamma + 1.0)
            rho = (
                rho_L * (p_star_over_p_L + gmoogpo) / (gmoogpo * p_star_over_p_L + 1.0)
            )
            u = u_star
            p = p_star

        return rho, u, p

    @staticmethod
    def sample_left_rarefaction_wave(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the left state for a rarefaction wave.

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
        a_L : float
            Sound speed of the left state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        # The rarefaction wave is enclosed by the Head and the Tail

        # Characteristics speed of the head
        S_HL = u_L - a_L

        # Left state regime
        if speed < S_HL:
            rho = rho_L
            u = u_L
            p = p_L

        # Rarefaction wave regime
        else:
            # Sound speed behind the rarefaction
            a_star_L = a_L * (p_star / p_L) ** ((gamma - 1.0) / (2.0 * gamma))

            # Characteristics speed of the tail
            S_TL = u_star - a_star_L

            # Rarefaction fan regime
            if speed < S_TL:
                two_over_gamma_plus_one = 2.0 / (gamma + 1.0)
                common = (
                    two_over_gamma_plus_one
                    + ((gamma - 1.0) / ((gamma + 1.0) * a_L)) * (u_L - speed)
                ) ** (2.0 / (gamma - 1.0))

                rho = rho_L * common
                u = two_over_gamma_plus_one * (
                    a_L + ((gamma - 1.0) / 2.0) * u_L + speed
                )
                p = p_L * common**gamma
            else:
                rho = rho_L * ((p_star / p_L) ** (1.0 / gamma))
                u = u_star
                p = p_star

        return rho, u, p

    @staticmethod
    def sample_right_shock_wave(
        gamma: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the right state for a shock wave.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_R : float
            Density of the right state.
        u_R : float
            Velocity of the right state.
        p_R : float
            Pressure of the right state.
        a_R : float
            Sound speed of the right state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        p_star_over_p_R = p_star / p_R

        # Shock speed
        S_R = u_R + a_R * math.sqrt(
            ((gamma + 1.0) / (2.0 * gamma)) * p_star_over_p_R
            + (gamma - 1.0) / (2.0 * gamma)
        )

        # Right state regime
        if speed > S_R:
            rho = rho_R
            u = u_R
            p = p_R

        # Middle state regime
        else:
            # Gamma minus one over gamma plus one
            gmoogpo = (gamma - 1.0) / (gamma + 1.0)
            rho = (
                rho_R * (p_star_over_p_R + gmoogpo) / (gmoogpo * p_star_over_p_R + 1.0)
            )
            u = u_star
            p = p_star

        return rho, u, p

    @staticmethod
    def sample_right_rarefaction_wave(
        gamma: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        u_star: float,
        p_star: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample the right state for a rarefaction wave.

        Parameters
        ----------
        gamma : float
            Adiabatic index.
        rho_R : float
            Density of the right state.
        u_R : float
            Velocity of the right state.
        p_R : float
            Pressure of the right state.
        a_R : float
            Sound speed of the right state.
        u_star : float
            Velocity in the star region.
        p_star : float
            Pressure in the star region.
        speed : float
            Speed S = x / t when sampling at (x, t).

        Returns
        -------
        rho : float
            Density in the middle state.
        u : float
            Velocity in the middle state.
        p : float
            Pressure in the middle state.
        """
        # The rarefaction wave is enclosed by the Head and the Tail

        # Characteristics speed of the head
        S_HR = u_R + a_R

        # Right state regime
        if speed > S_HR:
            rho = rho_R
            u = u_R
            p = p_R

        # Rarefaction wave regime
        else:
            # Sound speed behind the rarefaction
            a_star_R = a_R * (p_star / p_R) ** ((gamma - 1.0) / (2.0 * gamma))

            # Characteristics speed of the tail
            S_TR = u_star + a_star_R

            # Rarefaction fan regime
            if speed > S_TR:
                two_over_gamma_plus_one = 2.0 / (gamma + 1.0)
                common = (
                    two_over_gamma_plus_one
                    - ((gamma - 1.0) / ((gamma + 1.0) * a_R)) * (u_R - speed)
                ) ** (2.0 / (gamma - 1.0))

                rho = rho_R * common
                u = two_over_gamma_plus_one * (
                    -a_R + ((gamma - 1.0) / 2.0) * u_R + speed
                )
                p = p_R * common**gamma

            else:
                rho = rho_R * ((p_star / p_R) ** (1.0 / gamma))
                u = u_star
                p = p_star

        return rho, u, p
