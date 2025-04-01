/**
 * \file slope_limiter.h
 * 
 * \brief Slope limiter functions for MUSCL reconstruction.
 */

#ifndef LIMITER_H
#define LIMITER_H

#include "common.h"

#define LIMITER_MINMOD 1
#define LIMITER_VAN_LEER 2
#define LIMITER_MONOTONIZED_CENTER 3


/**
 * \brief Get the slope limiter flag.
 * 
 * This function sets the slope limiter flag in the integrator parameter.
 * 
 * \param[in, out] integrator_param Pointer to the integrator parameter.
 */
ErrorStatus get_slope_limiter_flag(
    IntegratorParam *__restrict integrator_param
);

/**
 * \brief Slope limiter function.
 * 
 * This function limits the slope using the specified limiter.
 * 
 * \param[out] limited_slope Pointer to the limited slope.
 * \param[in] integrator_param Pointer to the integrator parameter.
 * \param[in] slope_L Left slope.
 * \param[in] slope_R Right slope.
 * 
 * \return ErrorStatus
 * 
 * \exception VALUE_ERROR If the limiter is not recognized.
 */
ErrorStatus limit_slope(
    double *__restrict limited_slope,
    const IntegratorParam *__restrict integrator_param,
    const double slope_L,
    const double slope_R
);

#endif
