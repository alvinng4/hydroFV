/**
 * \file reconstruction.h
 * 
 * \brief Header file for reconstruction related definitions and prototypes.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-19
 */

#ifndef RECONSTRUCTION_H
#define RECONSTRUCTION_H

#include "common.h"
#include "system.h"

#define RECONSTRUCTION_PIECEWISE_CONSTANT 1
#define RECONSTRUCTION_PIECEWISE_LINEAR 2
#define RECONSTRUCTION_PIECEWISE_PARABOLIC 3

ErrorStatus get_reconstruction_flag(IntegratorParam *__restrict integrator_param);

void reconstruct_cell_interface(
    const IntegratorParam *__restrict integrator_param,
    const double *__restrict field,
    double *__restrict interface_field_L,
    double *__restrict interface_field_R,
    const int num_cells,
    const int num_ghost_cells_side
);

#endif
