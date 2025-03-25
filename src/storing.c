/**
 * \file storing.c
 * 
 * \brief Definitions of functions for storing snapshots of the hydrodynamic system.
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-25
 */

#include <hdf5.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* For mkdir */
#ifndef _WIN32
#include <sys/types.h>
#include <sys/stat.h>
#else
#include <direct.h>
#include <windows.h>
#endif

#include "hydro.h"

StoringParam get_new_storing_param(void)
{
    StoringParam storing_param = {
        .is_storing = false,
        .store_initial = false,
        .output_dir = NULL,
        .storing_interval = -1.0,
        .store_count_ = 0
    };

    return storing_param;
}

ErrorStatus finalize_storing_param(StoringParam *__restrict storing_param)
{
    /* Disable is disabled */
    if (!storing_param->is_storing)
    {
        return make_success_error_status();
    }

    /* Check storing interval */
    if (storing_param->storing_interval <= 0.0)
    {
        return WRAP_RAISE_ERROR(VALUE_ERROR, "Storing interval must be positive.");
    }

    /* Check directory path */
    if (!storing_param->output_dir)
    {
        // Set to snapshots_YYYYMMDD_HHMMSS if directory path is not set
        size_t path_str_len = strlen("snapshots_YYYYMMDD_HHMMSS/") + 1;
        char *output_dir = malloc(path_str_len * sizeof(char));
        if (!output_dir)
        {
            return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for directory path.");
        }
        const time_t raw_time = time(NULL);
        struct tm *time_info = localtime(&raw_time);
        strftime(output_dir, path_str_len, "snapshots_%Y%m%d_%H%M%S/", time_info);
        storing_param->output_dir = output_dir;
    }
    else
    {
        if (storing_param->output_dir[strlen(storing_param->output_dir) - 1] != '/')
        {
            size_t buffer_size = (
                strlen("Directory path for storing snapshots must end with a trailing slash (\"/\"). Got: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *error_message = malloc(buffer_size * sizeof(char));
            if (!error_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
            }
            snprintf(
                error_message,
                buffer_size,
                "Directory path for storing snapshots must end with a trailing slash (\"/\"). Got: \"%s\".",
                storing_param->output_dir
            );
            return WRAP_RAISE_ERROR(VALUE_ERROR, error_message);
        }
    }

    /* Create directory */
#ifdef _WIN32
    if (_mkdir(storing_param->output_dir) == -1)
    {
        if (GetFileAttributes(storing_param->output_dir) == INVALID_FILE_ATTRIBUTES)
        {
            size_t buffer_size = (
                strlen("Failed to access path for storing snapshots: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *error_message = malloc(buffer_size * sizeof(char));
            if (!error_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
            }
            snprintf(
                error_message,
                buffer_size,
                "Failed to access path for storing snapshots: \"%s\".",
                storing_param->output_dir
            );
            return WRAP_RAISE_ERROR(OS_ERROR, error_message);
        }
        else if (GetFileAttributes(storing_param->output_dir) & FILE_ATTRIBUTE_DIRECTORY)
        {
            int buffer_size = (
                strlen("Directory for storing snapshots already exists. The files will be overwritten. Directory: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *warning_message = malloc(buffer_size * sizeof(char));
            if (!warning_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for warning message.");
            }
            const int actual_warning_message_length = snprintf(
                warning_message,
                buffer_size,
                "Directory for storing snapshots already exists. The files will be overwritten. Directory: \"%s\".",
                storing_param->output_dir
            );
            if (actual_warning_message_length < 0)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Failed to get warning message string.");
            }
            else if (actual_warning_message_length >= buffer_size)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Warning message string is truncated.");
            }
            WRAP_RAISE_WARNING(warning_message);
        }
    }
#else
    struct stat st = {0};
    if (mkdir(storing_param->output_dir, 0777) == -1)
    {
        if(stat(storing_param->output_dir, &st) != 0)
        {
            size_t buffer_size = (
                strlen("Failed to access path for storing snapshots: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *error_message = malloc(buffer_size * sizeof(char));
            if (!error_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
            }
            snprintf(
                error_message,
                buffer_size,
                "Failed to access path for storing snapshots: \"%s\".",
                storing_param->output_dir
            );
            return WRAP_RAISE_ERROR(OS_ERROR, error_message);
        }

        else if (st.st_mode & S_IFDIR)
        {
            int buffer_size = (
                strlen("Directory for storing snapshots already exists. The files will be overwritten. Directory: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *warning_message = malloc(buffer_size * sizeof(char));
            if (!warning_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for warning message.");
            }
            const int actual_warning_message_length = snprintf(
                warning_message,
                buffer_size,
                "Directory for storing snapshots already exists. The files will be overwritten. Directory: \"%s\".",
                storing_param->output_dir
            );
            if (actual_warning_message_length < 0)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Failed to get warning message string.");
            }
            else if (actual_warning_message_length >= buffer_size)
            {
                return WRAP_RAISE_ERROR(VALUE_ERROR, "Warning message string is truncated.");
            }
            WRAP_RAISE_WARNING(warning_message);
        }
#endif

        else 
        {
            size_t buffer_size = (
                strlen("Failed to create directory for storing snapshots: \"\".")
                + strlen(storing_param->output_dir)
                + 1  // Null terminator
            );
            char *error_message = malloc(buffer_size * sizeof(char));
            if (!error_message)
            {
                return WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for error message.");
            }
            snprintf(error_message, buffer_size, "Failed to create directory for storing snapshots: \"%s\".", storing_param->output_dir);
            return WRAP_RAISE_ERROR(OS_ERROR, error_message);
        }
    }

    return make_success_error_status();
}

/**
 * \brief Store a snapshot of the 1D hydrodynamic system to an HDF5 file.
 * 
 * \param boundary_condition_param Pointer to the boundary condition parameters.
 * \param system Pointer to the 1D hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param simulation_status Pointer to the simulation status.
 * \param storing_param Pointer to the storing parameters.
 * \param file_path Path to the HDF5 file.
 */
IN_FILE ErrorStatus store_snapshot_1d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const SimulationStatus *__restrict simulation_status,
    StoringParam *__restrict storing_param,
    const char *__restrict file_path
)
{
    (void) integrator_param;

    ErrorStatus error_status;
    const hsize_t total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;

    /* Create HDF5 file */
    hid_t file = H5Fcreate(file_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 file.");
        goto err_create_hdf5_file;
    }

    /* Create HDF5 group */
    hid_t simulation_status_group = H5Gcreate(file, "/simulation_status", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t fields_group = H5Gcreate(file, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t parameters_group = H5Gcreate(file, "/parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (simulation_status_group == H5I_INVALID_HID || fields_group == H5I_INVALID_HID || parameters_group == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 group.");
        goto err_create_hdf5_group;
    }

    /* Create HDF5 dataspace */
    hid_t scaler_dataspace = H5Screate(H5S_SCALAR);
    hid_t field_dataspace = H5Screate_simple(1, &total_num_cells_x, NULL);
    if (scaler_dataspace == H5I_INVALID_HID || field_dataspace == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 dataspace.");
        goto err_create_dataspace;
    }

    /* Create HDF5 datatypes */
    hid_t variable_length_str = H5Tcopy(H5T_C_S1);
    H5Tset_size(variable_length_str, H5T_VARIABLE);

    /* Create simulation status datasets */
    hid_t dataset_num_steps = H5Dcreate(simulation_status_group, "num_steps", H5T_NATIVE_LLONG, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_simulation_time = H5Dcreate(simulation_status_group, "simulation_time", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_dt = H5Dcreate(simulation_status_group, "dt", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_num_steps == H5I_INVALID_HID || dataset_simulation_time == H5I_INVALID_HID || dataset_dt == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 simulation status datasets.");
        goto err_create_simulation_status_datasets;
    }

    /* Create fields datasets */
    hid_t dataset_density = H5Dcreate(fields_group, "density", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_velocity_x = H5Dcreate(fields_group, "velocity_x", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_pressure = H5Dcreate(fields_group, "pressure", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_density == H5I_INVALID_HID || dataset_velocity_x == H5I_INVALID_HID || dataset_pressure == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 fields datasets.");
        goto err_create_fields_datasets;
    }

    /* Create parameters datasets */
    hid_t dataset_x_min = H5Dcreate(parameters_group, "x_min", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_x_max = H5Dcreate(parameters_group, "x_max", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_num_cells_x = H5Dcreate(parameters_group, "num_cells_x", H5T_NATIVE_INT, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_num_ghost_cells_side = H5Dcreate(parameters_group, "num_ghost_cells_side", H5T_NATIVE_INT, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_dx = H5Dcreate(parameters_group, "dx", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_gamma = H5Dcreate(parameters_group, "gamma", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_coordinate_system = H5Dcreate(parameters_group, "coordinate_system", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_x_min = H5Dcreate(parameters_group, "boundary_condition_x_min", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_x_max = H5Dcreate(parameters_group, "boundary_condition_x_max", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (
        dataset_x_min == H5I_INVALID_HID
        || dataset_x_max == H5I_INVALID_HID
        || dataset_num_cells_x == H5I_INVALID_HID
        || dataset_num_ghost_cells_side == H5I_INVALID_HID
        || dataset_dx == H5I_INVALID_HID
        || dataset_gamma == H5I_INVALID_HID
        || dataset_coordinate_system == H5I_INVALID_HID
        || dataset_boundary_condition_x_min == H5I_INVALID_HID
        || dataset_boundary_condition_x_max == H5I_INVALID_HID
    )
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 parameters datasets.");
        goto err_create_parameters_datasets;
    }

    /* Write data to HDF5 dataset */
    H5Dwrite(dataset_num_steps, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->num_steps);
    H5Dwrite(dataset_simulation_time, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->t);
    H5Dwrite(dataset_dt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->dt);

    H5Dwrite(dataset_density, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->density_);
    H5Dwrite(dataset_velocity_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->velocity_x_);
    H5Dwrite(dataset_pressure, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->pressure_);
    H5Dwrite(dataset_x_min, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->x_min);
    H5Dwrite(dataset_x_max, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->x_max);
    H5Dwrite(dataset_num_cells_x, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->num_cells_x);
    H5Dwrite(dataset_num_ghost_cells_side, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->num_ghost_cells_side);
    H5Dwrite(dataset_dx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->dx_);
    H5Dwrite(dataset_gamma, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->gamma);
    H5Dwrite(dataset_coordinate_system, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(system->coord_sys));
    H5Dwrite(dataset_boundary_condition_x_min, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_x_min));
    H5Dwrite(dataset_boundary_condition_x_max, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_x_max));

    /* Close HDF5 dataset, dataspace, group, and file */
    H5Dclose(dataset_x_min);
    H5Dclose(dataset_x_max);
    H5Dclose(dataset_num_cells_x);
    H5Dclose(dataset_num_ghost_cells_side);
    H5Dclose(dataset_dx);
    H5Dclose(dataset_gamma);
    H5Dclose(dataset_coordinate_system);
    H5Dclose(dataset_boundary_condition_x_min);
    H5Dclose(dataset_boundary_condition_x_max);

    H5Dclose(dataset_num_steps);
    H5Dclose(dataset_simulation_time);
    H5Dclose(dataset_dt);

    H5Dclose(dataset_density);
    H5Dclose(dataset_velocity_x);
    H5Dclose(dataset_pressure);    

    H5Sclose(scaler_dataspace);
    H5Sclose(field_dataspace);

    H5Gclose(simulation_status_group);
    H5Gclose(fields_group);
    H5Gclose(parameters_group);

    H5Fclose(file);

    (storing_param->store_count_)++;

    return make_success_error_status();

err_create_parameters_datasets:
    if (dataset_x_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_x_min);
    }
    if (dataset_x_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_x_max);
    }
    if (dataset_num_cells_x != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_cells_x);
    }
    if (dataset_num_ghost_cells_side != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_ghost_cells_side);
    }
    if (dataset_dx != H5I_INVALID_HID)
    {
        H5Dclose(dataset_dx);
    }
    if (dataset_gamma != H5I_INVALID_HID)
    {
        H5Dclose(dataset_gamma);
    }
    if (dataset_coordinate_system != H5I_INVALID_HID)
    {
        H5Dclose(dataset_coordinate_system);
    }
    if (dataset_boundary_condition_x_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_x_min);
    }
    if (dataset_boundary_condition_x_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_x_max);
    }
err_create_fields_datasets:
    if (dataset_density != H5I_INVALID_HID)
    {
        H5Dclose(dataset_density);
    }
    if (dataset_velocity_x != H5I_INVALID_HID)
    {
        H5Dclose(dataset_velocity_x);
    }
    if (dataset_pressure != H5I_INVALID_HID)
    {
        H5Dclose(dataset_pressure);
    }
err_create_simulation_status_datasets:
    if (dataset_num_steps != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_steps);
    }
    if (dataset_simulation_time != H5I_INVALID_HID)
    {
        H5Dclose(dataset_simulation_time);
    }
    if (dataset_dt != H5I_INVALID_HID)
    {
        H5Dclose(dataset_dt);
    }
err_create_dataspace:
    if (scaler_dataspace != H5I_INVALID_HID)
    {
        H5Sclose(scaler_dataspace);
    }
    if (field_dataspace != H5I_INVALID_HID)
    {
        H5Sclose(field_dataspace);
    }
err_create_hdf5_group:
    if (simulation_status_group != H5I_INVALID_HID)
    {
        H5Gclose(simulation_status_group);
    }
    if (fields_group != H5I_INVALID_HID)
    {
        H5Gclose(fields_group);
    }
    if (parameters_group != H5I_INVALID_HID)
    {
        H5Gclose(parameters_group);
    }
err_create_hdf5_file:
    if (file != H5I_INVALID_HID)
    {
        H5Fclose(file);
    }
    return error_status;
}

/**
 * \brief Store a snapshot of the 2D hydrodynamic system to an HDF5 file.
 * 
 * \param boundary_condition_param Pointer to the boundary condition parameters.
 * \param system Pointer to the 2D hydrodynamic system.
 * \param integrator_param Pointer to the integrator parameters.
 * \param simulation_status Pointer to the simulation status.
 * \param storing_param Pointer to the storing parameters.
 * \param file_path Path to the HDF5 file.
 */
IN_FILE ErrorStatus store_snapshot_2d(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const SimulationStatus *__restrict simulation_status,
    StoringParam *__restrict storing_param,
    const char *__restrict file_path
)
{
    (void) integrator_param;

    ErrorStatus error_status;
    const hsize_t total_num_cells_x = system->num_cells_x + 2 * system->num_ghost_cells_side;
    const hsize_t total_num_cells_y = system->num_cells_y + 2 * system->num_ghost_cells_side;
    const hsize_t total_num_cells = total_num_cells_x * total_num_cells_y;

    /* Create HDF5 file */
    hid_t file = H5Fcreate(file_path, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (file == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 file.");
        goto err_create_hdf5_file;
    }

    /* Create HDF5 group */
    hid_t simulation_status_group = H5Gcreate(file, "/simulation_status", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t fields_group = H5Gcreate(file, "/fields", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t parameters_group = H5Gcreate(file, "/parameters", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (simulation_status_group == H5I_INVALID_HID || fields_group == H5I_INVALID_HID || parameters_group == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 group.");
        goto err_create_hdf5_group;
    }

    /* Create HDF5 dataspace */
    hid_t scaler_dataspace = H5Screate(H5S_SCALAR);
    hid_t field_dataspace = H5Screate_simple(1, &total_num_cells, NULL);
    if (scaler_dataspace == H5I_INVALID_HID || field_dataspace == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 dataspace.");
        goto err_create_dataspace;
    }

    /* Create HDF5 datatypes */
    hid_t variable_length_str = H5Tcopy(H5T_C_S1);
    H5Tset_size(variable_length_str, H5T_VARIABLE);

    /* Create simulation status datasets */
    hid_t dataset_num_steps = H5Dcreate(simulation_status_group, "num_steps", H5T_NATIVE_LLONG, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_simulation_time = H5Dcreate(simulation_status_group, "simulation_time", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_dt = H5Dcreate(simulation_status_group, "dt", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_num_steps == H5I_INVALID_HID || dataset_simulation_time == H5I_INVALID_HID || dataset_dt == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 simulation status datasets.");
        goto err_create_simulation_status_datasets;
    }

    /* Create fields datasets */
    hid_t dataset_density = H5Dcreate(fields_group, "density", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_velocity_x = H5Dcreate(fields_group, "velocity_x", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_velocity_y = H5Dcreate(fields_group, "velocity_y", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_pressure = H5Dcreate(fields_group, "pressure", H5T_NATIVE_DOUBLE, field_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    if (dataset_density == H5I_INVALID_HID || dataset_velocity_x == H5I_INVALID_HID || dataset_pressure == H5I_INVALID_HID)
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 fields datasets.");
        goto err_create_fields_datasets;
    }

    /* Create parameters datasets */
    hid_t dataset_x_min = H5Dcreate(parameters_group, "x_min", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_x_max = H5Dcreate(parameters_group, "x_max", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_y_min = H5Dcreate(parameters_group, "y_min", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_y_max = H5Dcreate(parameters_group, "y_max", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_num_cells_x = H5Dcreate(parameters_group, "num_cells_x", H5T_NATIVE_INT, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_num_cells_y = H5Dcreate(parameters_group, "num_cells_y", H5T_NATIVE_INT, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_num_ghost_cells_side = H5Dcreate(parameters_group, "num_ghost_cells_side", H5T_NATIVE_INT, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_dx = H5Dcreate(parameters_group, "dx", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_dy = H5Dcreate(parameters_group, "dy", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_gamma = H5Dcreate(parameters_group, "gamma", H5T_NATIVE_DOUBLE, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_coordinate_system = H5Dcreate(parameters_group, "coordinate_system", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_x_min = H5Dcreate(parameters_group, "boundary_condition_x_min", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_x_max = H5Dcreate(parameters_group, "boundary_condition_x_max", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_y_min = H5Dcreate(parameters_group, "boundary_condition_y_min", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    hid_t dataset_boundary_condition_y_max = H5Dcreate(parameters_group, "boundary_condition_y_max", variable_length_str, scaler_dataspace, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

    if (
        dataset_x_min == H5I_INVALID_HID
        || dataset_x_max == H5I_INVALID_HID
        || dataset_y_min == H5I_INVALID_HID
        || dataset_y_max == H5I_INVALID_HID
        || dataset_num_cells_x == H5I_INVALID_HID
        || dataset_num_cells_y == H5I_INVALID_HID
        || dataset_num_ghost_cells_side == H5I_INVALID_HID
        || dataset_dx == H5I_INVALID_HID
        || dataset_dy == H5I_INVALID_HID
        || dataset_gamma == H5I_INVALID_HID
        || dataset_coordinate_system == H5I_INVALID_HID
        || dataset_boundary_condition_x_min == H5I_INVALID_HID
        || dataset_boundary_condition_x_max == H5I_INVALID_HID
        || dataset_boundary_condition_y_min == H5I_INVALID_HID
        || dataset_boundary_condition_y_max == H5I_INVALID_HID
    )
    {
        error_status = WRAP_RAISE_ERROR(OS_ERROR, "Failed to create HDF5 parameters datasets.");
        goto err_create_parameters_datasets;
    }

    /* Write data to HDF5 dataset */
    H5Dwrite(dataset_num_steps, H5T_NATIVE_LLONG, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->num_steps);
    H5Dwrite(dataset_simulation_time, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->t);
    H5Dwrite(dataset_dt, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &simulation_status->dt);

    H5Dwrite(dataset_density, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->density_);
    H5Dwrite(dataset_velocity_x, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->velocity_x_);
    H5Dwrite(dataset_velocity_y, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->velocity_y_);
    H5Dwrite(dataset_pressure, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, system->pressure_);

    H5Dwrite(dataset_x_min, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->x_min);
    H5Dwrite(dataset_x_max, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->x_max);
    H5Dwrite(dataset_y_min, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->y_min);
    H5Dwrite(dataset_y_max, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->y_max);
    H5Dwrite(dataset_num_cells_x, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->num_cells_x);
    H5Dwrite(dataset_num_cells_y, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->num_cells_y);
    H5Dwrite(dataset_num_ghost_cells_side, H5T_NATIVE_INT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->num_ghost_cells_side);
    H5Dwrite(dataset_dx, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->dx_);
    H5Dwrite(dataset_dy, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->dy_);
    H5Dwrite(dataset_gamma, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &system->gamma);
    H5Dwrite(dataset_coordinate_system, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(system->coord_sys));
    H5Dwrite(dataset_boundary_condition_x_min, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_x_min));
    H5Dwrite(dataset_boundary_condition_x_max, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_x_max));
    H5Dwrite(dataset_boundary_condition_y_min, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_y_min));
    H5Dwrite(dataset_boundary_condition_y_max, variable_length_str, H5S_ALL, H5S_ALL, H5P_DEFAULT, &(boundary_condition_param->boundary_condition_y_max));

    /* Close HDF5 dataset, dataspace, group, and file */
    H5Dclose(dataset_x_min);
    H5Dclose(dataset_x_max);
    H5Dclose(dataset_y_min);
    H5Dclose(dataset_y_max);
    H5Dclose(dataset_num_cells_x);
    H5Dclose(dataset_num_cells_y);
    H5Dclose(dataset_num_ghost_cells_side);
    H5Dclose(dataset_dx);
    H5Dclose(dataset_dy);
    H5Dclose(dataset_gamma);
    H5Dclose(dataset_coordinate_system);
    H5Dclose(dataset_boundary_condition_x_min);
    H5Dclose(dataset_boundary_condition_x_max);
    H5Dclose(dataset_boundary_condition_y_min);
    H5Dclose(dataset_boundary_condition_y_max);

    H5Dclose(dataset_density);
    H5Dclose(dataset_velocity_x);
    H5Dclose(dataset_velocity_y);
    H5Dclose(dataset_pressure);  
    
    H5Dclose(dataset_num_steps);
    H5Dclose(dataset_simulation_time);
    H5Dclose(dataset_dt);

    H5Sclose(scaler_dataspace);
    H5Sclose(field_dataspace);

    H5Gclose(simulation_status_group);
    H5Gclose(fields_group);
    H5Gclose(parameters_group);

    H5Fclose(file);

    (storing_param->store_count_)++;

    return make_success_error_status();

err_create_parameters_datasets:
    if (dataset_x_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_x_min);
    }
    if (dataset_x_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_x_max);
    }
    if (dataset_y_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_y_min);
    }
    if (dataset_y_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_y_max);
    }
    if (dataset_num_cells_x != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_cells_x);
    }
    if (dataset_num_cells_y != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_cells_y);
    }
    if (dataset_num_ghost_cells_side != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_ghost_cells_side);
    }
    if (dataset_dx != H5I_INVALID_HID)
    {
        H5Dclose(dataset_dx);
    }
    if (dataset_dy != H5I_INVALID_HID)
    {
        H5Dclose(dataset_dy);
    }
    if (dataset_gamma != H5I_INVALID_HID)
    {
        H5Dclose(dataset_gamma);
    }
    if (dataset_coordinate_system != H5I_INVALID_HID)
    {
        H5Dclose(dataset_coordinate_system);
    }
    if (dataset_boundary_condition_x_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_x_min);
    }
    if (dataset_boundary_condition_x_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_x_max);
    }
    if (dataset_boundary_condition_y_min != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_y_min);
    }
    if (dataset_boundary_condition_y_max != H5I_INVALID_HID)
    {
        H5Dclose(dataset_boundary_condition_y_max);
    }
err_create_fields_datasets:
    if (dataset_density != H5I_INVALID_HID)
    {
        H5Dclose(dataset_density);
    }
    if (dataset_velocity_x != H5I_INVALID_HID)
    {
        H5Dclose(dataset_velocity_x);
    }
    if (dataset_velocity_y != H5I_INVALID_HID)
    {
        H5Dclose(dataset_velocity_y);
    }
    if (dataset_pressure != H5I_INVALID_HID)
    {
        H5Dclose(dataset_pressure);
    }
err_create_simulation_status_datasets:
    if (dataset_num_steps != H5I_INVALID_HID)
    {
        H5Dclose(dataset_num_steps);
    }
    if (dataset_simulation_time != H5I_INVALID_HID)
    {
        H5Dclose(dataset_simulation_time);
    }
    if (dataset_dt != H5I_INVALID_HID)
    {
        H5Dclose(dataset_dt);
    }
err_create_dataspace:
    if (scaler_dataspace != H5I_INVALID_HID)
    {
        H5Sclose(scaler_dataspace);
    }
    if (field_dataspace != H5I_INVALID_HID)
    {
        H5Sclose(field_dataspace);
    }
err_create_hdf5_group:
    if (simulation_status_group != H5I_INVALID_HID)
    {
        H5Gclose(simulation_status_group);
    }
    if (fields_group != H5I_INVALID_HID)
    {
        H5Gclose(fields_group);
    }
    if (parameters_group != H5I_INVALID_HID)
    {
        H5Gclose(parameters_group);
    }
err_create_hdf5_file:
    if (file != H5I_INVALID_HID)
    {
        H5Fclose(file);
    }
    return error_status;
}


ErrorStatus store_snapshot(
    const BoundaryConditionParam *__restrict boundary_condition_param,
    const System *__restrict system,
    const IntegratorParam *__restrict integrator_param,
    const SimulationStatus *__restrict simulation_status,
    StoringParam *__restrict storing_param
)
{
    ErrorStatus error_status;

    if (!storing_param->output_dir)
    {
        error_status = WRAP_RAISE_ERROR(POINTER_ERROR, "Directory path for storing is NULL.");
        goto err_output_dir_null;
    }

    /* Make file path string */
    const int file_path_length = (
        strlen(storing_param->output_dir)
        + snprintf(NULL, 0, "snapshot_%d.h5", storing_param->store_count_)
        + 1  // Null terminator
    );
    char *file_path = malloc(file_path_length * sizeof(char));
    if (!file_path)
    {
        error_status = WRAP_RAISE_ERROR(MEMORY_ERROR, "Failed to allocate memory for file path string.");
        goto err_file_path_memory_alloc;
    }
    int actual_file_path_length = snprintf(file_path, file_path_length, "%ssnapshot_%d.h5", storing_param->output_dir, storing_param->store_count_);

    if (actual_file_path_length < 0)
    {
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Failed to get storing file path string");
        goto err_write_file_path_string;
    }
    else if (actual_file_path_length >= file_path_length)
    {
        error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Storing file path string is truncated.");
        goto err_write_file_path_string;
    }

    /* Store snapshot */
    switch (system->coord_sys_flag_)
    {
        case COORD_SYS_CARTESIAN_1D: case COORD_SYS_CYLINDRICAL_1D: case COORD_SYS_SPHERICAL_1D:
            error_status = store_snapshot_1d(boundary_condition_param, system, integrator_param, simulation_status, storing_param, file_path);
            if (error_status.return_code != SUCCESS)
            {
                goto err_store_snapshot;
            }
            break;
        case COORD_SYS_CARTESIAN_2D:
            error_status = store_snapshot_2d(boundary_condition_param, system, integrator_param, simulation_status, storing_param, file_path);
            if (error_status.return_code != SUCCESS)
            {
                goto err_store_snapshot;
            }
            break;
        case COORD_SYS_CARTESIAN_3D:
            error_status = WRAP_RAISE_ERROR(NOT_IMPLEMENTED_ERROR, "Storing snapshot for Cartesian 3D system is not implemented.");
            goto err_store_snapshot;
        default:
            error_status = WRAP_RAISE_ERROR(VALUE_ERROR, "Coordinate system flag not recognized.");
            goto err_store_snapshot;
    }

    free(file_path);

    return make_success_error_status();

err_store_snapshot:
err_write_file_path_string:
err_file_path_memory_alloc:
    free(file_path);
err_output_dir_null:
    return error_status;
}
