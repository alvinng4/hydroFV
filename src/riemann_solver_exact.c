/**
 * \file riemann_solver_exact.c
 * 
 * \brief Exact Riemann solver for the 1D Euler equations.
 * 
 * \cite Toro, E. F., Riemann Solvers and Numerical Methods for Fluid Dynamics,
 *       3rd ed. Springer., 2009.
 * \cite Press, W. H., et al., "Bracketing and Bisection" in Numerical Recipes 
 *       in C: The Art of Scientific Computing, 2nd ed.
 *       Cambridge University Press, 1992, pp. 350-354.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-08
 */

#include <math.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hydro.h"
#include "riemann_solver.h"
#include "utils.h"

#define NEWTON_RAPHSON_MAX_ITER 100
#define BISECTION_BRACKET_NUM_INTERVALS 100
#define BISECTION_MAX_ITER 5000

/**
 * \brief Riemann f_L or f_R function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_X Density of the left or right state.
 * \param p_X Pressure of the left or right state.
 * \param a_X Sound speed of the left or right state.
 * \param p_star Pressure at the star regime.
 * 
 * \return The value of the Riemann f_L or f_R function.
 */
IN_FILE real riemann_f_L_or_R(
    const real gamma,
    const real rho_X,
    const real p_X,
    const real a_X,
    const real p_star
);

/**
 * \brief Derivative of the Riemann f_L or f_R function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_X Density of the left or right state.
 * \param p_X Pressure of the left or right state.
 * \param a_X Sound speed of the left or right state.
 * \param p_star Pressure at the star regime.
 * 
 * \return The value of the derivative of the Riemann f_L or f_R function.
 */
IN_FILE real riemann_f_L_or_R_prime(
    const real gamma,
    const real rho_X,
    const real p_X,
    const real a_X,
    const real p_star
);

/**
 * \brief Riemann f function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state. 
 * \param a_R Sound speed of the right state.
 * \param p_star Pressure at the star regime.
 * 
 * \return The value of the Riemann f function.
 */
IN_FILE real riemann_f(
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real p_star
);

/**
 * \brief Derivative of the Riemann f function.
 * 
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param rho_R Density of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param p_star Pressure at the star regime.
 * 
 * \return The value of the derivative of the Riemann f function.
 */
IN_FILE real riemann_f_prime(
    const real gamma,
    const real rho_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real p_R,
    const real a_R,
    const real p_star
);

/**
 * \brief Solve for the pressure at the star regime, using
 *        a hybrid of Newton-Raphson and bisection methods.
 * 
 * \param p_star Pointer to the pressure at the star regime.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param tol Tolerance.
 * \param verbose Verbosity level.
 * 
 * \retval SUCCESS if successful
 * \retval ERROR_RIEMANN_SOLVER_EXACT_P_STAR_BISECTION_BRACKET_ROOT if failed to bracket the root
 * \retval ERROR_RIEMANN_SOLVER_EXACT_P_STAR_NOT_CONVERGED if p_star did not converge after maximum iterations
 */
IN_FILE ErrorStatus solve_p_star(
    real *restrict p_star,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real tol,
    const int verbose
);

/**
 * \brief Sample the Riemann problem solution for the left state.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_left_state(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
);

/**
 * \brief Sample the Riemann problem solution for the right state.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_right_state(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
);

/**
 * \brief Sample the Riemann problem solution for the left shock wave.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_left_shock_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
);

/**
 * \brief Sample the Riemann problem solution for the left rarefaction wave.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_L Density of the left state.
 * \param u_L Velocity of the left state.
 * \param p_L Pressure of the left state.
 * \param a_L Sound speed of the left state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_left_rarefaction_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
);

/**
 * \brief Sample the Riemann problem solution for the right shock wave.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_right_shock_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
);

/**
 * \brief Sample the Riemann problem solution for the right rarefaction wave.
 * 
 * \param sol_rho Pointer to the solution density.
 * \param sol_u Pointer to the solution velocity.
 * \param sol_p Pointer to the solution pressure.
 * \param gamma Adiabatic index.
 * \param rho_R Density of the right state.
 * \param u_R Velocity of the right state.
 * \param p_R Pressure of the right state.
 * \param a_R Sound speed of the right state.
 * \param u_star Velocity at the star regime.
 * \param p_star Pressure at the star regime.
 * \param speed Speed S = x / t when sampling at (x, t).
 */
IN_FILE void sample_right_rarefaction_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
);

ErrorStatus solve_exact(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real tol,
    const real speed,
    const int verbose
)
{
    ErrorStatus error_status;

    /* Compute the sound speeds */
    real a_L;
    real a_R;
    if (rho_L > 0.0)
    {
        a_L = get_sound_speed(gamma, rho_L, p_L);
    }
    else
    {
        // Vacuum
        // Assuming isentropic-type EOS, p = p(rho), see Toro [1]
        a_L = 0.0;
    }

    if (rho_R > 0.0)
    {
        a_R = get_sound_speed(gamma, rho_R, p_R);
    }
    else
    {
        // Vacuum
        // Assuming isentropic-type EOS, p = p(rho), see Toro [1]
        a_R = 0.0;
    }

    /**
     * Check for vacuum or vacuum generation
     * (1) If left or right states is vacuum, or
     * (2) pressure positivity condition is met
     */
    if (
        rho_L <= 0.0 ||
        rho_R <= 0.0 ||
        ((a_L + a_R) * 2.0 / (gamma + 1.0)) <= (u_R - u_L)
    )
    {
        solve_vacuum(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            rho_R,
            u_R,
            p_R,
            a_R,
            speed
        );
        goto exit_success;
    }
    
    /* Solve for p_star and u_star */
    real p_star;
    error_status = WRAP_TRACEBACK(solve_p_star(
        &p_star,
        gamma,
        rho_L,
        u_L,
        p_L,
        a_L,
        rho_R,
        u_R,
        p_R,
        a_R,
        tol,
        verbose
    ));
    if (error_status.return_code != SUCCESS)
    {
        goto err_solve_p_star;
    }

    real u_star = 0.5 * (u_L + u_R) + 0.5 * (
        riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
        - riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
    );

    /* The riemann problem is solved. Now we sample the solution */
    if (speed < u_star)
    {
        sample_left_state(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            u_star,
            p_star,
            speed
        );
    }
    else
    {
        sample_right_state(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_R,
            u_R,
            p_R,
            a_R,
            u_star,
            p_star,
            speed
        );
    }

exit_success:
    return make_success_error_status();

err_solve_p_star:
    return error_status;
}

ErrorStatus solve_flux_exact(
    real *restrict flux_mass,
    real *restrict flux_momentum,
    real *restrict flux_energy,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real tol,
    const real speed,
    const int verbose
)
{
    real sol_rho;
    real sol_u;
    real sol_p;
    ErrorStatus error_status = WRAP_TRACEBACK(solve_exact(
        &sol_rho,
        &sol_u,
        &sol_p,
        gamma,
        rho_L,
        u_L,
        p_L,
        rho_R,
        u_R,
        p_R,
        tol,
        speed,
        verbose
    ));

    *flux_mass = sol_rho * sol_u;
    *flux_momentum = sol_rho * sol_u * sol_u + sol_p;
    *flux_energy = (sol_p * (gamma / (gamma - 1.0)) + 0.5 * sol_rho * sol_u * sol_u) * sol_u;
    return error_status;
}

IN_FILE real riemann_f_L_or_R(
    const real gamma,
    const real rho_X,
    const real p_X,
    const real a_X,
    const real p_star
)
{
    /* shock wave */
    if (p_star > p_X)
    {
        real A_X = riemann_A_L_or_R(gamma, rho_X);
        real B_X = riemann_B_L_or_R(gamma, p_X);
        return (p_star - p_X) * sqrt(A_X / (p_star + B_X));
    }
    /* rarefaction wave */
    else
    {
        return (
            2.0
            * a_X
            * (pow((p_star / p_X), (gamma - 1.0) / (2.0 * gamma)) - 1.0) 
            / (gamma - 1.0)
        );
    }
}

IN_FILE real riemann_f_L_or_R_prime(
    const real gamma,
    const real rho_X,
    const real p_X,
    const real a_X,
    const real p_star
)
{
    /* shock wave */
    if (p_star > p_X)
    {
        real A_X = riemann_A_L_or_R(gamma, rho_X);
        real B_X = riemann_B_L_or_R(gamma, p_X);
        return (
            (1.0 - 0.5 * (p_star - p_X) / (B_X - p_star))
            * sqrt(A_X / (B_X + p_star))
        );
    }
    /* rarefaction wave */
    else
    {
        return pow(
            p_star / p_X,
            -0.5 * (gamma + 1.0) / gamma
        ) / (p_X * a_X);
    }
}

IN_FILE real riemann_f(
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real p_star
)
{
    return (
        riemann_f_L_or_R(gamma, rho_L, p_L, a_L, p_star)
        + riemann_f_L_or_R(gamma, rho_R, p_R, a_R, p_star)
        + (u_R - u_L)
    );
}

IN_FILE real riemann_f_prime(
    const real gamma,
    const real rho_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real p_R,
    const real a_R,
    const real p_star
)
{
    return (
        riemann_f_L_or_R_prime(gamma, rho_L, p_L, a_L, p_star)
        + riemann_f_L_or_R_prime(gamma, rho_R, p_R, a_R, p_star)
    );
}

IN_FILE ErrorStatus solve_p_star(
    real *restrict p_star,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real tol,
    const int verbose
)
{
    ErrorStatus error_status;

    real p_guess = guess_p(
        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, tol
    );

    // For bisection method
    bool bracket_found = false;
    real p_upper_bisection = p_guess;
    real f_upper_bisection = riemann_f(
        gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_upper_bisection
    );
    real p_lower_bisection;
    real f_lower_bisection;

    /* Newton-Raphson method */
    real p_0 = p_guess;
    for (int i = 0; i < NEWTON_RAPHSON_MAX_ITER; i++)
    {
        real f = riemann_f(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_0
        );
        real f_prime = riemann_f_prime(
            gamma, rho_L, p_L, a_L, rho_R, p_R, a_R, p_0
        );
        if (!bracket_found && f * f_upper_bisection < 0.0)
        {
            bracket_found = true;
            f_lower_bisection = f;
            p_lower_bisection = p_0;
        }

        real p_1 = p_0 - f / f_prime;

        /* Failed to converge, switch to bisection method */
        if (p_1 < 0.0)
        {
            break;
        }

        if (2.0 * fabs(p_1 - p_0) / (p_1 + p_0) < tol)
        {
            *p_star = p_1;
            goto exit_success;
        }

        p_0 = p_1;
    }

    /* Bisection method */
    // Find the lower bound and upper bound
    if (!bracket_found)
    {
        p_lower_bisection = 0.0;
        f_lower_bisection = riemann_f(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_lower_bisection
        );
        if (f_lower_bisection * f_upper_bisection >= 0.0)
        {
            p_upper_bisection *= 10.0;
            real dp = (p_upper_bisection - p_lower_bisection) / BISECTION_BRACKET_NUM_INTERVALS;
            for (int i = 0; i < BISECTION_BRACKET_NUM_INTERVALS; i++)
            {
                real _p_lower = p_lower_bisection + i * dp;
                real _p_upper = p_lower_bisection + (i + 1) * dp;

                real _f_lower = riemann_f(
                    gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, _p_lower
                );
                real _f_upper = riemann_f(
                    gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, _p_upper
                );

                if (_f_lower * _f_upper < 0.0)
                {
                    p_lower_bisection = _p_lower;
                    f_lower_bisection = _f_lower;
                    p_upper_bisection = _p_upper;
                    f_upper_bisection = _f_upper;
                    break;
                }
            }
            
            // Failed to bracket the root
            if (f_lower_bisection * f_upper_bisection >= 0.0)
            {
                error_status = WRAP_RAISE_ERROR(FAILURE, "Failed to bracket the root for bisection method.");
                goto err_bisection_bracket_root;
            }
        }
    }

    int count = 0;
    while (true)
    {
        real p_mid = 0.5 * (p_lower_bisection + p_upper_bisection);
        if (p_upper_bisection - p_lower_bisection < tol)
        {
            *p_star = p_mid;
            break;
        }
        real f_mid = riemann_f(
            gamma, rho_L, u_L, p_L, a_L, rho_R, u_R, p_R, a_R, p_mid
        );

        if (f_mid * f_lower_bisection < 0.0)
        {
            p_upper_bisection = p_mid;
            f_upper_bisection = f_mid;
        }
        else
        {
            p_lower_bisection = p_mid;
            f_lower_bisection = f_mid;
        }

        count++;
        if (count % 500 == 0 && verbose > 0)
        {
            const size_t buffer_size = strlen(
                "Bisection method did not converge after  iterations for exact riemann solver."
            ) + 256;
            char buffer[buffer_size];
            snprintf(buffer, buffer_size, "Bisection method did not converge after %d iterations for exact riemann solver.", count);
            WRAP_RAISE_WARNING(buffer);
        }
        if (count > BISECTION_MAX_ITER)
        {
            const size_t buffer_size = strlen(
                "Bisection method did not converge after  iterations for exact riemann solver."
            ) + 256;
            char buffer[buffer_size];
            snprintf(buffer, buffer_size, "Bisection method did not converge after %d iterations for exact riemann solver.", count);
            error_status = WRAP_RAISE_ERROR(FAILURE, buffer);
            goto err_converge;
        }
    }

exit_success:
    return make_success_error_status();

err_converge:
err_bisection_bracket_root:
    return error_status;
}

IN_FILE void sample_left_state(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
)
{
    /* Shock wave */
    if (p_star > p_L)
    {
        sample_left_shock_wave(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            u_star,
            p_star,
            speed
        );
        return;
    }
    /* Rarefaction wave */
    else
    {
        sample_left_rarefaction_wave(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_L,
            u_L,
            p_L,
            a_L,
            u_star,
            p_star,
            speed
        );
        return;
    }
}

IN_FILE void sample_right_state(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
)
{
    /* Shock wave */
    if (p_star > p_R)
    {
        sample_right_shock_wave(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_R,
            u_R,
            p_R,
            a_R,
            u_star,
            p_star,
            speed
        );
        return;
    }
    /* Rarefaction wave */
    else
    {
        sample_right_rarefaction_wave(
            sol_rho,
            sol_u,
            sol_p,
            gamma,
            rho_R,
            u_R,
            p_R,
            a_R,
            u_star,
            p_star,
            speed
        );
        return;
    }
}

IN_FILE void sample_left_shock_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
)
{
    real p_star_over_p_L = p_star / p_L;

    // Shock speed
    real S_L = u_L - a_L * sqrt(
        (gamma + 1.0) / (2.0 * gamma) * p_star_over_p_L
        + (gamma - 1.0) / (2.0 * gamma)
    );

    /* Left state regime */
    if (speed < S_L)
    {
        *sol_rho = rho_L;
        *sol_u = u_L;
        *sol_p = p_L;
        return;
    }

    /* Star regime */
    else
    {
        // Gamma minus one divided by gamma plus one
        real gmodgpo = (gamma - 1.0) / (gamma + 1.0);
        *sol_rho = (
            rho_L * (p_star_over_p_L + gmodgpo) / (gmodgpo * p_star_over_p_L + 1.0)
        );
        *sol_u = u_star;
        *sol_p = p_star;
        return;
    }
}

IN_FILE void sample_left_rarefaction_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_L,
    const real u_L,
    const real p_L,
    const real a_L,
    const real u_star,
    const real p_star,
    const real speed
)
{
    // The rarefaction wave is enclosed by the Head and the Tail

    // Characteristics speed of the head
    real S_HL = u_L - a_L;

    /* Left state regime */
    if (speed < S_HL)
    {
        *sol_rho = rho_L;
        *sol_u = u_L;
        *sol_p = p_L;
        return;
    }

    // Sound speed behind the rarefaction
    real a_star_L = a_L * pow(p_star / p_L, (gamma - 1.0) / (2.0 * gamma));

    // Characteristics speed of the tail
    real S_TL = u_star - a_star_L;

    /* Rarefaction fan regime */
    if (speed < S_TL)
    {
        // Two divided by gamma plus one
        real tdgpo = 2.0 / (gamma + 1.0);
        real common = pow(
            (
                tdgpo
                + (u_L - speed) * ((gamma - 1.0) / (a_L * (gamma + 1.0)))
            ), 2.0 / (gamma - 1.0)
        );

        *sol_rho = rho_L * common;
        *sol_u = tdgpo * (a_L + 0.5 * u_L * (gamma - 1.0) + speed);
        *sol_p = p_L * pow(common, gamma);
        return;
    }

    /* Star regime */
    else
    {
        *sol_rho = rho_L * pow(p_star / p_L, 1.0 / gamma);
        *sol_u = u_star;
        *sol_p = p_star;
        return;
    }
}

IN_FILE void sample_right_shock_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
)
{
    real p_star_over_p_R = p_star / p_R;

    // Shock speed
    real S_R = u_R + a_R * sqrt(
        p_star_over_p_R * (gamma + 1.0) / (2.0 * gamma)
        + (gamma - 1.0) / (2.0 * gamma)
    );

    /* Right state regime */
    if (speed > S_R)
    {
        *sol_rho = rho_R;
        *sol_u = u_R;
        *sol_p = p_R;
        return;
    }

    /* Star regime */
    else
    {
        // Gamma minus one divided by gamma plus one
        real gmodgpo = (gamma - 1.0) / (gamma + 1.0);
        *sol_rho = (
            rho_R * (p_star_over_p_R + gmodgpo) / (gmodgpo * p_star_over_p_R + 1.0)
        );
        *sol_u = u_star;
        *sol_p = p_star;
        return;
    }
}

IN_FILE void sample_right_rarefaction_wave(
    real *restrict sol_rho,
    real *restrict sol_u,
    real *restrict sol_p,
    const real gamma,
    const real rho_R,
    const real u_R,
    const real p_R,
    const real a_R,
    const real u_star,
    const real p_star,
    const real speed
)
{
    // The rarefaction wave is enclosed by the Head and the Tail

    // Characteristics speed of the head
    real S_HR = u_R + a_R;

    /* Right state regime */
    if (speed > S_HR)
    {
        *sol_rho = rho_R;
        *sol_u = u_R;
        *sol_p = p_R;
        return;
    }

    // Sound speed behind the rarefaction
    real a_star_R = a_R * pow(p_star / p_R, (gamma - 1.0) / (2.0 * gamma));

    // Characteristics speed of the tail
    real S_TR = u_star + a_star_R;

    /* Rarefaction fan regime */
    if (speed > S_TR)
    {
        real two_divided_by_gamma_plus_one = 2.0 / (gamma + 1.0);
        real common = pow(
            (
                two_divided_by_gamma_plus_one
                - (u_R - speed) * (gamma - 1.0) / (a_R * (gamma + 1.0))
            ), 2.0 / (gamma - 1.0)
        );

        *sol_rho = rho_R * common;
        *sol_u = two_divided_by_gamma_plus_one * (-a_R + 0.5 * u_R * (gamma - 1.0) + speed);
        *sol_p = p_R * pow(common, gamma);
        return;
    }

    /* Star regime */
    else
    {
        *sol_rho = rho_R * pow(p_star / p_R, 1.0 / gamma);
        *sol_u = u_star;
        *sol_p = p_star;
        return;
    }
}
