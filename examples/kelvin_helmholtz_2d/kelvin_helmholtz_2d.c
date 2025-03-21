#include <math.h>
#include <stdio.h>
#include <time.h>

#include "hydro.h"

#define RIEMANN_SOLVER "riemann_solver_hllc"
#define COORD_SYS "cartesian_2d"
#define NUM_TOTAL_CELLS_X 512
#define NUM_TOTAL_CELLS_Y 512
#define NUM_GHOST_CELLS_SIDE 3
#define NUM_CELLS_X NUM_TOTAL_CELLS_X - 2 * NUM_GHOST_CELLS_SIDE
#define NUM_CELLS_Y NUM_TOTAL_CELLS_Y - 2 * NUM_GHOST_CELLS_SIDE
#define INTEGRATOR "godunov_first_order_2d"
#define RECONSTRUCTION "piecewise_parabolic" // "piecewise_constant", "piecewise_linear" or "piecewise_parabolic"
#define LIMITER "monotonized_center" // "minmod", "van_leer" or "monotonized_center"
#define TIME_INTEGRATOR "ssp_rk3" // "euler", "ssp_rk2" or "ssp_rk3"

#define CFL 0.4
#define TF 20.0
#define TOL 1e-6 // For the riemann solver

/* Parameters */
#define GAMMA 5.0 / 3.0
#define LEFT_BOUNDARY_CONDITION "periodic"
#define RIGHT_BOUNDARY_CONDITION "periodic"
#define BOTTOM_BOUNDARY_CONDITION "periodic"
#define TOP_BOUNDARY_CONDITION "periodic"
#define DOMAIN_MIN_X 0.0
#define DOMAIN_MAX_X 1.0
#define DOMAIN_MIN_Y 0.0
#define DOMAIN_MAX_Y 1.0
#define RHO_1 1.0
#define RHO_2 2.0
#define RHO_M 0.5 * (RHO_1 - RHO_2)
#define U_1 0.5
#define U_2 -0.5
#define U_M 0.5 * (U_1 - U_2)
#define L 0.025


IN_FILE ErrorStatus get_initial_system(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system
)
{
    ErrorStatus error_status;
    if (system->coord_sys_flag_ != COORD_SYS_CARTESIAN_2D)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Invalid coordinate system, Expected Cartesian 2D.");
    }

    const int num_ghost_cells_side = system->num_ghost_cells_side;
    const int num_cells_x = system->num_cells_x;
    const int num_cells_y = system->num_cells_y;
    const int total_num_cells_x = num_cells_x + 2 * num_ghost_cells_side;

    double *mid_points_x = system->mid_points_x_;
    double *mid_points_y = system->mid_points_y_;
    double *density = system->density_;
    double *velocity_x = system->velocity_x_;
    double *velocity_y = system->velocity_y_;
    double *pressure = system->pressure_;

    for (int i = num_ghost_cells_side; i < (num_ghost_cells_side + num_cells_x); i++)
    {
        for (int j = num_ghost_cells_side; j < (num_ghost_cells_side + num_cells_y); j++)
        {
            const int idx = j * total_num_cells_x + i;
            if (mid_points_y[j] < 0.25)
            {
                density[idx] = RHO_1 - RHO_M * exp((mid_points_y[j] - 0.25) / L);
                velocity_x[idx] = U_1 - U_M * exp((mid_points_y[j] - 0.25) / L);
            }
            else if (mid_points_y[j] < 0.5)
            {
                density[idx] = RHO_2 + RHO_M * exp((- mid_points_y[j] + 0.25) / L);
                velocity_x[idx] = U_2 + U_M * exp((- mid_points_y[j] + 0.25) / L);
            }
            else if (mid_points_y[j] < 0.75)
            {
                density[idx] = RHO_2 + RHO_M * exp((mid_points_y[j] - 0.75) / L);
                velocity_x[idx] = U_2 + U_M * exp((mid_points_y[j] - 0.75) / L);
            }
            else
            {
                density[idx] = RHO_1 - RHO_M * exp((- mid_points_y[j] + 0.75) / L);
                velocity_x[idx] = U_1 - U_M * exp((- mid_points_y[j] + 0.75) / L);
            }

            velocity_y[idx] = 0.01 * sin(4.0 * M_PI * mid_points_x[i]);
            pressure[idx] = 2.5;
        }
    }

    boundary_condition_param->boundary_condition_x_min = LEFT_BOUNDARY_CONDITION;
    boundary_condition_param->boundary_condition_x_max = RIGHT_BOUNDARY_CONDITION;
    boundary_condition_param->boundary_condition_y_min = BOTTOM_BOUNDARY_CONDITION;
    boundary_condition_param->boundary_condition_y_max = TOP_BOUNDARY_CONDITION;

    error_status = finalize_boundary_condition_param(system, boundary_condition_param);
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }

    error_status = WRAP_TRACEBACK(set_boundary_condition(boundary_condition_param, system));
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }
    error_status = WRAP_TRACEBACK(convert_primitive_to_conserved(system));
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }

    return make_success_error_status();
}

int main(void)
{
    ErrorStatus error_status;

    /* Boundary conditions */
    BoundaryConditionParam boundary_condition_param = get_new_boundary_condition_param();

    /* System parameters */
    System system = get_new_system_struct();
    system.coord_sys = COORD_SYS;
    system.gamma = GAMMA;
    system.x_min = DOMAIN_MIN_X;
    system.x_max = DOMAIN_MAX_X;
    system.y_min = DOMAIN_MIN_Y;
    system.y_max = DOMAIN_MAX_Y;
    system.num_cells_x = NUM_CELLS_X;
    system.num_cells_y = NUM_CELLS_Y;
    system.num_ghost_cells_side = NUM_GHOST_CELLS_SIDE;

    error_status = WRAP_TRACEBACK(system_init(&system));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    error_status = WRAP_TRACEBACK(get_initial_system(&boundary_condition_param, &system));
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }

    /* Integrator parameters */
    IntegratorParam integrator_param = get_new_integrator_param();
    integrator_param.integrator = INTEGRATOR;
    integrator_param.riemann_solver = RIEMANN_SOLVER;
    integrator_param.reconstruction = RECONSTRUCTION;
    integrator_param.reconstruction_limiter = LIMITER;
    integrator_param.time_integrator = TIME_INTEGRATOR;
    integrator_param.cfl = CFL;

    /* Storing parameters */
    StoringParam storing_param = get_new_storing_param();
    storing_param.is_storing = true;
    storing_param.store_initial = true;
    storing_param.storing_interval = 0.05;
    storing_param.output_dir = "snapshots/";

    /* Settings */
    Settings settings = {
        .verbose = 1,
        .no_progress_bar = false
    };

    /* Simulation parameters */
    SimulationParam simulation_param = {
        .tf = TF
    };

    /* Simulation status */
    SimulationStatus simulation_status;

    printf("Launching simulation...\n");
    time_t start = clock();
    error_status = WRAP_TRACEBACK(launch_simulation(
        &boundary_condition_param,
        &system,
        &integrator_param,
        &storing_param,
        &settings,
        &simulation_param,
        &simulation_status
    ));
    time_t end = clock();
    if (error_status.return_code != SUCCESS)
    {
        goto error;
    }
    printf("Simulation time: %f s, Number of steps: %lld\n", (double) (end - start) / CLOCKS_PER_SEC, simulation_status.num_steps);
    printf("Done!\n");

    free_system_memory(&system);
    return 0;

error:
    print_and_free_traceback(&error_status);
    return 1;
}
