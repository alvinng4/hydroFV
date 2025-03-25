/**
 * \file common.h
 * 
 * \brief Common definitions for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-20
 */

#ifndef COMMON_H
#define COMMON_H

#include <stdbool.h>
#include <stdint.h>

/* Functions that are only used in the same file */
#define IN_FILE static


typedef int64_t int64;

typedef struct ErrorStatus
{
    int return_code;
    char *traceback;
    int traceback_code_;
} ErrorStatus;

typedef struct IntegratorParam
{
    char *integrator;
    char *riemann_solver;
    char *reconstruction;
    char *reconstruction_limiter;
    char *time_integrator;
    double cfl;
    double cfl_initial_shrink_factor;
    int cfl_initial_shrink_num_steps;
    double tol;

    int integrator_flag_;
    int riemann_solver_flag_;
    int reconstruction_flag_;
    int reconstruction_limiter_flag_;
    int time_integrator_flag_;
} IntegratorParam;

typedef struct StoringParam
{
    char *output_dir;
    bool is_storing;
    bool store_initial;
    double storing_interval;

    int storing_method_flag_;
    int store_count_;
} StoringParam;

typedef struct Settings
{
    int verbose;
    bool no_progress_bar;
} Settings;

typedef struct SimulationParam
{
    double tf;
} SimulationParam;

typedef struct SimulationStatus
{
    int64 num_steps;
    double t;
    double dt;
} SimulationStatus;

#endif
