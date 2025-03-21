#include <stdio.h>
#include <time.h>

#include "hydro.h"

#define RIEMANN_SOLVER "riemann_solver_hllc" // "riemann_solver_exact" or "riemann_solver_hllc"
#define COORD_SYS "spherical_1d" // "cartesian_1d", "cylindrical_1d" or "spherical_1d"
#define NUM_TOTAL_CELLS 128
#define NUM_GHOST_CELLS_SIDE 3
#define NUM_CELLS NUM_TOTAL_CELLS - 2 * NUM_GHOST_CELLS_SIDE
#define INTEGRATOR "godunov_first_order_1d" // "godunov_first_order_1d" or "random_choice_1d"
#define RECONSTRUCTION "piecewise_parabolic" // "piecewise_constant", "piecewise_linear" or "piecewise_parabolic"
#define LIMITER "monotonized_center" // "minmod", "van_leer" or "monotonized_center"
#define TIME_INTEGRATOR "ssp_rk3" // "euler", "ssp_rk2" or "ssp_rk3"

#define CFL 0.4
#define TF 1.0
#define TOL 1e-6 // For the riemann solver

/* Sedov Blast parameters */
#define NUM_EXPLOSION_CELLS 12

#define GAMMA 1.4
#define RHO_0 1.0
#define U_0 0.0
#define P_0 1e-5
#define LEFT_BOUNDARY_CONDITION "reflective"
#define RIGHT_BOUNDARY_CONDITION "transmissive"
#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.2

IN_FILE ErrorStatus get_initial_system(
    BoundaryConditionParam *__restrict boundary_condition_param,
    System *__restrict system
)
{
    ErrorStatus error_status;

    boundary_condition_param->boundary_condition_x_min = LEFT_BOUNDARY_CONDITION;
    boundary_condition_param->boundary_condition_x_max = RIGHT_BOUNDARY_CONDITION;
    error_status = finalize_boundary_condition_param(system, boundary_condition_param);
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }
    
    /* Initial condition */
    double E_0;
    if (system->coord_sys_flag_ == COORD_SYS_CARTESIAN_1D)
    {
        E_0 = 0.0673185 / NUM_EXPLOSION_CELLS;
    }
    else if (system->coord_sys_flag_ == COORD_SYS_CYLINDRICAL_1D)
    {
        E_0 = 0.311357 / NUM_EXPLOSION_CELLS;
    }
    else if (system->coord_sys_flag_ == COORD_SYS_SPHERICAL_1D)
    {
        E_0 = 0.851072 / NUM_EXPLOSION_CELLS;
    }
    else
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Invalid coordinate system");
    }

    for (int i = system->num_ghost_cells_side; i < (system->num_ghost_cells_side + system->num_cells_x); i++)
    {
        system->density_[i] = RHO_0;
        system->velocity_x_[i] = U_0;
        system->pressure_[i] = P_0;
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

    for (int i = NUM_GHOST_CELLS_SIDE; i < (NUM_GHOST_CELLS_SIDE + NUM_EXPLOSION_CELLS); i++)
    {
        system->energy_[i] = E_0;
    }

    error_status = WRAP_TRACEBACK(convert_conserved_to_primitive(system));
    if (error_status.return_code != SUCCESS)
    {
        return error_status;
    }
    error_status = WRAP_TRACEBACK(set_boundary_condition(boundary_condition_param, system));
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
    system.x_min = DOMAIN_MIN;
    system.x_max = DOMAIN_MAX;
    system.num_cells_x = NUM_CELLS;
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
    storing_param.store_initial = false;
    storing_param.storing_interval = TF; // Only store the final snapshot
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
