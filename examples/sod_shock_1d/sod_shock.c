#include <stdio.h>
#include "hydro.h"

#define RIEMANN_SOLVER "riemann_solver_hllc" // "riemann_solver_exact" or "riemann_solver_hllc"
#define COORD_SYS "cartesian_1d" // "cartesian_1d", "cylinrical_1d" or "spherical_1d"
#define NUM_TOTAL_CALLS 10000
#define NUM_GHOST_CELLS_SIDE 1
#define NUM_CELLS NUM_TOTAL_CALLS - 2 * NUM_GHOST_CELLS_SIDE
#define INTEGRATOR "godunov_first_order_1d" // "godunov_first_order_1d" or "random_choice_1d"

#define CFL 0.9
#define INITIAL_CFL_SHRINK_FACTOR 0.2
#define NUM_STEPS_SHRINK 50
#define TF 0.2
#define TOL 1e-6 // For the riemann solver

/* Sod shock parameters */
#define GAMMA 1.4
#define RHO_L 1.0
#define U_L 0.0
#define P_L 1.0
#define RHO_R 0.125
#define U_R 0.0
#define P_R 0.1
#define DISCONTINUITY_POS 0.5
#define LEFT_BOUNDARY_CONDITION "transmissive"
#define RIGHT_BOUNDARY_CONDITION "transmissive"
#define DOMAIN_MIN 0.0
#define DOMAIN_MAX 1.0

int main(void)
{
    int return_code;

    System system = {
        .coord_sys = COORD_SYS,
        .boundary_condition_x_min = LEFT_BOUNDARY_CONDITION,
        .boundary_condition_x_max = RIGHT_BOUNDARY_CONDITION,
        .gamma = GAMMA,
        .x_min = DOMAIN_MIN,
        .x_max = DOMAIN_MAX,
        .num_cells_x = NUM_CELLS,
        .num_ghost_cells_side = NUM_GHOST_CELLS_SIDE
    };

    return_code = system_init(&system);
    if (return_code != SUCCESS)
    {
        goto error;
    }

    const int total_num_cells = system.num_cells_x + 2 * system.num_ghost_cells_side;
    for (int i = 0; i < total_num_cells; i++)
    {
        if (system.mid_points_x_[i] < DISCONTINUITY_POS)
        {
            system.density_[i] = RHO_L;
            system.velocity_[i] = U_L;
            system.pressure_[i] = P_L;
        }
        else
        {
            system.density_[i] = RHO_R;
            system.velocity_[i] = U_R;
            system.pressure_[i] = P_R;
        }
    }
    convert_primitive_to_conserved(&system);

    IntegratorParam integrator_param = {
        .integrator = INTEGRATOR,
        .riemann_solver = RIEMANN_SOLVER,
        .cfl = CFL,
        .cfl_initial_shrink_factor = INITIAL_CFL_SHRINK_FACTOR,
        .num_steps_shrink = NUM_STEPS_SHRINK,
        .tol = TOL
    };

    StoringParam storing_param;

    Settings settings = {
        .verbose = 1
    };

    SimulationParam simulation_param = {
        .tf = TF
    };

    return_code = launch_simulation(
        &system,
        &integrator_param,
        &storing_param,
        &settings,
        &simulation_param
    );
    if (return_code != SUCCESS)
    {
        goto error;
    }

    FILE *file = fopen("sod_shock_1d.csv", "w");
    fprintf(file, "mid_point,density,velocity,pressure\n");
    for (int i = NUM_GHOST_CELLS_SIDE; i < (NUM_GHOST_CELLS_SIDE + NUM_CELLS); i++)
    {
        fprintf(
            file,
            "%f,%f,%f,%f\n",
            system.mid_points_x_[i],
            system.density_[i],
            system.velocity_[i],
            system.pressure_[i]
        );
    }

    free_system_memory(&system);
    return 0;

error:
    print_error_msg(return_code);
    return 1;
}