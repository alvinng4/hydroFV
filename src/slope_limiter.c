/**
 * \file slope_limiter.c
 * 
 * \brief Definitions of slope limiter functions for MUSCL reconstruction.
 * 
 * \author Ching-Yin Ng
 * \date April 2025
 */

#include <math.h>
#include <string.h>

#include "common.h"
#include "error.h"
#include "integrator.h"
#include "slope_limiter.h"

ErrorStatus get_slope_limiter_flag(
    IntegratorParam *__restrict integrator_param
)
{
    if (!integrator_param ||
        !integrator_param->slope_limiter
    )
    {
        return WRAP_RAISE_ERROR(POINTER_ERROR, "Integrator parameter or slope limiter is NULL.");
    }

    if (strcmp(integrator_param->slope_limiter, "minmod") == 0)
    {
        integrator_param->slope_limiter_flag_ = SLOPE_LIMITER_MINMOD;
    }
    else if (strcmp(integrator_param->slope_limiter, "van_leer") == 0)
    {
        integrator_param->slope_limiter_flag_ = SLOPE_LIMITER_VAN_LEER;
    }
    else if (strcmp(integrator_param->slope_limiter, "monotonized_center") == 0)
    {
        integrator_param->slope_limiter_flag_ = SLOPE_LIMITER_MONOTONIZED_CENTRAL;
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Slope limiter not recognized.");
    }

    return make_success_error_status();
}

IN_FILE double minmod(const double a, const double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else if (fabs(a) < fabs(b))
    {
        return a;
    }
    else
    {
        return b;
    }
}

IN_FILE double van_leer(const double a, const double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else
    {
        return 2.0 * a * b / (a + b + 1e-10);
    }
}

IN_FILE double monotonized_central(const double a, const double b)
{
    if (a * b <= 0.0)
    {
        return 0.0;
    }
    else if (a + b > 0)
    {
        return fmin(0.5 * (a + b), 2.0 * fmin(fabs(a), fabs(b)));
    }
    else
    {
        return -fmin(-0.5 * (a + b), 2.0 * fmin(fabs(a), fabs(b)));
    }
}

ErrorStatus limit_slope(
    double *__restrict limited_slope,
    const IntegratorParam *__restrict integrator_param,
    const double slope_L,
    const double slope_R
)
{
    switch (integrator_param->slope_limiter_flag_)
    {
        case SLOPE_LIMITER_MINMOD:
            *limited_slope = minmod(slope_L, slope_R);
            break;
        case SLOPE_LIMITER_VAN_LEER:
            *limited_slope = van_leer(slope_L, slope_R);
            break;
        case SLOPE_LIMITER_MONOTONIZED_CENTRAL:
            *limited_slope = monotonized_central(slope_L, slope_R);
            break;
        default:
            return WRAP_RAISE_ERROR(VALUE_ERROR, "Limiter not recognized.");
    }
    return make_success_error_status();
}
