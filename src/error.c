/**
 * \file error.c
 * 
 * \brief Exception handling functions for the hydrodynamics simulation
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hydro.h"

#define RESET "\033[0m"
#define BRIGHT_RED "\033[1;31m"
#define DIM_RED "\033[5;31;40m"
#define YELLOW_BOLD "\033[1;33m"
#define CYAN_REGULAR "\033[0;36m"
#define PURPLE_REGULAR "\033[0;35m"
#define PURPLE_BRIGHT "\033[0;95m"
#define PURPLE_BRIGHT_BOLD "\033[1;95m"

ErrorStatus make_success_error_status(void)
{
    ErrorStatus error_status = {
        .return_code = SUCCESS,
        .traceback = NULL,
        .traceback_code_ = TRACEBACK_NOT_INITIALIZED
    };
    return error_status;
}

void raise_warning(
    const char *__restrict warning_msg,
    const char *__restrict warning_file,
    const int warning_line,
    const char *__restrict warning_func
)
{
    fprintf(
        stderr,
        "%sWarning:%s In %s\"%s\"%s, line %s%d%s in %s%s%s:\n    %s%s%s\n",
        YELLOW_BOLD,
        RESET,
        CYAN_REGULAR,
        warning_file,
        RESET,
        CYAN_REGULAR,
        warning_line,
        RESET,
        CYAN_REGULAR,
        warning_func,
        RESET,
        PURPLE_REGULAR,
        warning_msg,
        RESET
    );
}

ErrorStatus raise_error(
    const int error_code,
    const char *__restrict error_msg,
    const char *__restrict error_file,
    const int error_line,
    const char *__restrict error_func
)
{
    ErrorStatus error_status = {
        .traceback = NULL,
        .traceback_code_ = TRACEBACK_NOT_INITIALIZED
    };

    char *error_type;
    switch (error_code)
    {
        case FAILURE:
            error_status.return_code = error_code;
            error_type = "Failure";
            break;
        case ERROR:
            error_status.return_code = error_code;
            error_type = "Error";
            break;
        case VALUE_ERROR:
            error_status.return_code = error_code;
            error_type = "ValueError";
            break;
        case POINTER_ERROR:
            error_status.return_code = error_code;
            error_type = "PointerError";
            break;
        case MEMORY_ERROR:
            error_status.return_code = error_code;
            error_type = "MemoryError";
            break;
        case OS_ERROR:
            error_status.return_code = error_code;
            error_type = "OSError";
            break;
        case NOT_IMPLEMENTED_ERROR:
            error_status.return_code = error_code;
            error_type = "NotImplementedError";
            break;
        default:
            error_status.return_code = UNKNOWN_ERROR;
            error_type = "UnknownError";
            break;
    }

    const int traceback_size = (
        strlen(error_file)
        + strlen(error_func)
        + strlen(error_msg)
        + strlen(error_type)
        + 3 * strlen(CYAN_REGULAR)
        + strlen(PURPLE_BRIGHT_BOLD)
        + strlen(PURPLE_REGULAR)
        + 5 * strlen(RESET)
        + strlen("    File \"\", line  in \n: \n")
        + snprintf(NULL, 0, "%d", error_line)   // Number of digits in error_line
        + 1  // Null terminator
    );
    error_status.traceback = malloc(traceback_size * sizeof(char));
    if (!error_status.traceback)
    {
        error_status.traceback_code_ = TRACEBACK_MALLOC_FAILED;
        free(error_status.traceback);
        error_status.traceback = NULL;

        goto err_memory_alloc;
    }

    int actual_traceback_size = snprintf(
        error_status.traceback,
        traceback_size,
        "    File %s\"%s\"%s, line %s%d%s in %s%s%s\n%s%s%s: %s%s%s\n",
        CYAN_REGULAR,
        error_file,
        RESET,
        CYAN_REGULAR,
        error_line,
        RESET,
        CYAN_REGULAR,
        error_func,
        RESET,
        PURPLE_BRIGHT_BOLD,
        error_type,
        RESET,
        PURPLE_REGULAR,
        error_msg,
        RESET
    );

    if (actual_traceback_size < 0)
    {
        error_status.traceback_code_ = TRACEBACK_SNPRINTF_FAILED;
        free(error_status.traceback);
        error_status.traceback = NULL;
    }
    else if (actual_traceback_size >= traceback_size)
    {
        error_status.traceback_code_ = TRACEBACK_TRUNCATED;
    }
    else
    {
        error_status.traceback_code_ = TRACEBACK_SUCCESS;
    }

    return error_status;

err_memory_alloc:
    return error_status;
}

ErrorStatus traceback(
    ErrorStatus error_status,
    const char *__restrict function_call_source_code,
    const char *__restrict error_file,
    const int error_line,
    const char *__restrict error_func
)
{
    if (
        error_status.return_code == SUCCESS
        || error_status.traceback_code_ != TRACEBACK_SUCCESS
    )
    {
        return error_status;
    }

    const int traceback_size = (
        strlen(error_file)
        + strlen(error_func)
        + strlen(function_call_source_code)
        + 3 * strlen(CYAN_REGULAR)
        + strlen(PURPLE_REGULAR)
        + 4 * strlen(RESET)
        + strlen("    File \"\", line  in \n: \n        \n")
        + snprintf(NULL, 0, "%d", error_line)   // Number of digits in error_line
        + strlen(error_status.traceback)
        + 1  // Null terminator
    );

    char *new_traceback = malloc(traceback_size * sizeof(char));
    if (!new_traceback)
    {
        error_status.traceback_code_ = TRACEBACK_MALLOC_FAILED;
        error_status.traceback = NULL;
        free(new_traceback);
        free(error_status.traceback);
        return error_status;
    }

    int actual_traceback_size = snprintf(
        new_traceback,
        traceback_size,
        "    File %s\"%s\"%s, line %s%d%s in %s%s%s\n        %s%s%s\n%s",
        CYAN_REGULAR,
        error_file,
        RESET,
        CYAN_REGULAR,
        error_line,
        RESET,
        CYAN_REGULAR,
        error_func,
        RESET,
        PURPLE_REGULAR,
        function_call_source_code,
        RESET,
        error_status.traceback
    );

    if (actual_traceback_size < 0)
    {
        error_status.traceback_code_ = TRACEBACK_SNPRINTF_FAILED;
        free(new_traceback);
        free(error_status.traceback);
        error_status.traceback = NULL;
    }
    else if (actual_traceback_size >= traceback_size)
    {
        free(error_status.traceback);
        error_status.traceback = new_traceback;
        error_status.traceback_code_ = TRACEBACK_TRUNCATED;
    }
    else
    {
        free(error_status.traceback);
        error_status.traceback = new_traceback;
        error_status.traceback_code_ = TRACEBACK_SUCCESS;
    }

    return error_status;
}

void free_traceback(ErrorStatus *__restrict error_status)
{
    if (error_status->traceback)
    {
        free(error_status->traceback);
        error_status->traceback = NULL;
    }
}

void print_and_free_traceback(ErrorStatus *__restrict error_status)
{
    fprintf(stderr, "%sTraceback%s %s(most recent call last):%s\n", BRIGHT_RED, RESET, DIM_RED, RESET);
    switch (error_status->traceback_code_)
    {
        case TRACEBACK_NOT_INITIALIZED:
            fputs("    Something went wrong. Traceback not initialized.\n", stderr);
            break;
        case TRACEBACK_SUCCESS:
            fputs(error_status->traceback, stderr);
            free(error_status->traceback);
            error_status->traceback = NULL;
            break;
        case TRACEBACK_MALLOC_FAILED:
            fputs("    Something went wrong. Failed to allocate memory for traceback.\n", stderr);
            break;
        case TRACEBACK_TRUNCATED:
            fputs(error_status->traceback, stderr);
            fputs("\n    Something went wrong. Traceback was truncated.\n", stderr);
            free(error_status->traceback);
            error_status->traceback = NULL;
            break;
        case TRACEBACK_SNPRINTF_FAILED:
            fputs("    Something went wrong. Failed to write to traceback.\n", stderr);
            break;
    }
}
