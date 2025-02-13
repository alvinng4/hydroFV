import math
import warnings
from typing import Tuple

import numpy as np

from . import utils


class ExactRiemannSolver:
    @staticmethod
    def solve_system(
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
        np.ndarray
            Density in the middle state.
        np.ndarray
            Pressure in the middle state.
        np.ndarray
            Velocity in the middle state.

        References
        ----------
        1. Toro, E. F., "The Riemann Problem for the Euler Equations" in
           Riemann Solvers and Numerical Methods for Fluid Dynamics,
           3rd ed. Springer., 2009, pp. 115-162.
        """
        if gamma <= 1.0:
            raise ValueError("The adiabatic index needs to be larger than 1!")

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
                sol_rho[i], sol_u[i], sol_p[i] = ExactRiemannSolver.solve_vacuum(
                    gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
                )
                continue

            ### Solve for p_star and u_star ###
            p_star = ExactRiemannSolver.solve_p_star(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
            )
            u_star = 0.5 * (u_L + u_R) + 0.5 * (
                ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
                - ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
            )

            ### The riemann problem is solved. Now we sample the solution ###
            if speed < u_star:
                sol_rho[i], sol_u[i], sol_p[i] = ExactRiemannSolver.sample_left_state(
                    gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
                )
            else:
                sol_rho[i], sol_u[i], sol_p[i] = ExactRiemannSolver.sample_right_state(
                    gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
                )

        return sol_rho, sol_u, sol_p

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

        References
        ----------
        1. Toro, E. F., "The Riemann Problem for the Euler Equations" in
           Riemann Solvers and Numerical Methods for Fluid Dynamics,
           3rd ed. Springer., 2009, pp. 115-162.
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
            rho, u, p = ExactRiemannSolver.solve_vacuum(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
            )
            return rho, u, p

        ### Solve for p_star and u_star ###
        p_star = ExactRiemannSolver.solve_p_star(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )
        u_star = 0.5 * (u_L + u_R) + 0.5 * (
            ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
            - ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
        )

        ### The riemann problem is solved. Now we sample the solution ###
        if speed < u_star:
            rho, u, p = ExactRiemannSolver.sample_left_state(
                gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
            )
        else:
            rho, u, p = ExactRiemannSolver.sample_right_state(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
            )

        return rho, u, p

    @staticmethod
    def solve_vacuum(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        # If both states are vacuum, then the solution is also vacuum
        if rho_L <= 0.0 and rho_R <= 0.0:
            return 0.0, 0.0, 0.0

        # Right state is vacuum
        if rho_L <= 0.0:
            rho, u, p = ExactRiemannSolver.sample_for_right_vacuum(
                gamma, rho_L, u_L, p_L, a_L, speed
            )

        # Left state is vacuum
        elif rho_R <= 0.0:
            rho, u, p = ExactRiemannSolver.sample_for_left_vacuum(
                gamma, rho_L, u_L, p_L, a_L, speed
            )

        # Vacuum generation
        else:
            rho, u, p = ExactRiemannSolver.sample_vacuum_generation(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, speed
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        p_guess = ExactRiemannSolver.guess_p(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
        )

        count = 0
        while True:
            f = ExactRiemannSolver.riemann_f(
                gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_guess
            )
            f_prime = ExactRiemannSolver.riemann_f_prime(
                gamma, rho_L, p_L, a_L, rho_R, p_R, a_R, p_guess
            )
            p = p_guess - f / f_prime

            if p < 0.0:
                raise ValueError("Negative pressure in the middle state!")

            if (abs(p - p_guess) / abs(0.5 * (p + p_guess))) < tol:
                return p

            p_guess = p

            count += 1

            if count > 1000:
                warnings.warn(
                    "Newton-Raphson method did not converge for 1000 iterations!"
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        if p_star > p_L:
            # Shock wave
            return ExactRiemannSolver.sample_left_shock_wave(
                gamma, rho_L, u_L, p_L, a_L, u_star, p_star, speed
            )
        else:
            # Rarefaction wave
            return ExactRiemannSolver.sample_left_rarefaction_wave(
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        if p_star > p_R:
            # Shock wave
            return ExactRiemannSolver.sample_right_shock_wave(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
            )
        else:
            # Rarefaction wave
            return ExactRiemannSolver.sample_right_rarefaction_wave(
                gamma, rho_R, u_R, p_R, a_R, u_star, p_star, speed
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        return ((gamma - 1.0) / (gamma + 1.0)) * p_X

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        # if p > p_X (shock)
        if p > p_X:
            A_X = ExactRiemannSolver.riemann_A_L_or_R(gamma, rho_X)
            B_X = ExactRiemannSolver.riemann_B_L_or_R(gamma, p_X)
            return (p - p_X) * np.sqrt(A_X / (p + B_X))

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        # if p > p_X (shock)
        if p > p_X:
            A_X = ExactRiemannSolver.riemann_A_L_or_R(gamma, rho_X)
            B_X = ExactRiemannSolver.riemann_B_L_or_R(gamma, p_X)
            return np.sqrt(A_X / (B_X + p)) * (1.0 - 0.5 * (p - p_X) / (B_X + p))

        # if p <= p_X (rarefaction)
        else:
            return (p / p_X) ** (-0.5 * (gamma + 1.0) / gamma) / (p_X * a_X)

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        return (
            ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p)
            + ExactRiemannSolver.riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p)
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
        return ExactRiemannSolver.riemann_f_L_or_R_prime(
            gamma, rho_L, p_L, a_L, p
        ) + ExactRiemannSolver.riemann_f_L_or_R_prime(gamma, rho_R, p_R, a_R, p)

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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
            A_L = ExactRiemannSolver.riemann_A_L_or_R(gamma, rho_L)
            B_L = ExactRiemannSolver.riemann_B_L_or_R(gamma, p_L)
            g_L = math.sqrt(A_L / (ppv + B_L))

            A_R = ExactRiemannSolver.riemann_A_L_or_R(gamma, rho_R)
            B_R = ExactRiemannSolver.riemann_B_L_or_R(gamma, p_R)
            g_R = math.sqrt(A_R / (ppv + B_R))

            p_guess = (g_L * p_L + g_R * p_R - (u_R - u_L)) / (g_L + g_R)

        # Prevent negative value
        return max(tol, p_guess)

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
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

    @staticmethod
    def sample_for_right_vacuum(
        gamma: float, rho_L: float, u_L: float, p_L: float, a_L: float, speed: float
    ) -> Tuple[float, float, float]:
        """Sample when the right state is vacuum.

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        ### Left state regime ###
        if speed <= u_L - a_L:
            rho = rho_L
            u = u_L
            p = p_L

        ### Vacuum regime ###
        else:
            # Speed of the front
            S_star_L = u_L + 2.0 * a_L / (gamma - 1.0)
            ### Rarefaction wave regime ###
            if speed < S_star_L:
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

            ### Vacuum regime ###
            else:
                rho = 0.0
                u = 0.0
                p = 0.0

        return rho, u, p

    @staticmethod
    def sample_for_left_vacuum(
        gamma: float, rho_R: float, u_R: float, p_R: float, a_R: float, speed: float
    ) -> Tuple[float, float, float]:
        """Sample when the left state is vacuum.

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        ### Right state regime ###
        if speed > u_R + a_R:
            rho = rho_R
            u = u_R
            p = p_R

        ### Vacuum regime ###
        else:
            # Speed of the front
            S_star_R = u_R - 2.0 * a_R / (gamma - 1.0)

            ### Rarefaction wave regime ###
            if speed > S_star_R:
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

            ### Vacuum regime ###
            else:
                rho = 0.0
                u = 0.0
                p = 0.0

        return rho, u, p

    @staticmethod
    def sample_vacuum_generation(
        gamma: float,
        rho_L: float,
        u_L: float,
        p_L: float,
        a_L: float,
        rho_R: float,
        u_R: float,
        p_R: float,
        a_R: float,
        speed: float,
    ) -> Tuple[float, float, float]:
        """Sample for vacuum generation.

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

        References
        ----------
        Toro, E. F., "The Riemann Problem for the Euler Equations" in
        Riemann Solvers and Numerical Methods for Fluid Dynamics,
        3rd ed. Springer., 2009, pp. 115-162.
        """
        # Speed of the left and right rarefaction waves
        S_star_L = 2.0 * a_L / (gamma - 1.0)
        S_star_R = 2.0 * a_R / (gamma - 1.0)

        # Left state
        if speed < S_star_L:
            rho, u, p = ExactRiemannSolver.sample_for_right_vacuum(
                gamma, rho_L, u_L, p_L, a_L, speed
            )

        elif speed < S_star_R:
            rho = 0.0
            u = 0.0
            p = 0.0

        else:
            rho, u, p = ExactRiemannSolver.sample_for_left_vacuum(
                gamma, rho_R, u_R, p_R, a_R, speed
            )

        return rho, u, p
