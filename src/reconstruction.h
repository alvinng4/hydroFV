/**
 * \file reconstruction.h
 * 
 * \brief Header file for reconstruction related definitions and prototypes.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-18
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "common.h"
#include "system.h"

#define RECONSTRUCTION_PIECEWISE_CONSTANT 1
#define RECONSTRUCTION_PIECEWISE_LINEAR 2
#define RECONSTRUCTION_PIECEWISE_PARABOLIC 3

ErrorStatus get_reconstruction_flag(IntegratorParam *__restrict integrator_param);

ErrorStatus reconstruct_cell_interface_1d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param
);

ErrorStatus reconstruct_cell_interface_2d(
    System *__restrict system,
    IntegratorParam *__restrict integrator_param
);

#endif
