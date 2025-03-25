/**
 * \file error.h
 * \brief Error codes and prototypes of error-related functions
 * 
 * This file contains error codes and prototypes of error-related functions for the C library. 
 * 
 * \author Ching-Yin Ng
 * \date 2025-03-11
 */

#ifndef ERROR_H
#define ERROR_H

/* Error codes */
#define UNKNOWN_ERROR -1
#define SUCCESS 0
#define FAILURE 1
#define VALUE_ERROR 2
#define POINTER_ERROR 3
#define MEMORY_ERROR 4
#define OS_ERROR 5
#define NOT_IMPLEMENTED_ERROR 6


/* Traceback code */
#define TRACEBACK_NOT_INITIALIZED -1
#define TRACEBACK_SUCCESS 0
#define TRACEBACK_MALLOC_FAILED 1
#define TRACEBACK_TRUNCATED 2
#define TRACEBACK_SNPRINTF_FAILED 3

/**
 * \brief Wrapper for raise_warning function.
 * 
 * \param error_msg Error message.
 */
#define WRAP_RAISE_WARNING(error_msg) \
    raise_warning(error_msg, __FILE__, __LINE__, __func__)

/**
 * \brief Wrapper for raise_error function.
 * 
 * \param error_status ErrorStatus struct.
 * \param error_code Error code.
 * \param error_msg Error message.
 * 
 * \return ErrorStatus struct.
 */
#define WRAP_RAISE_ERROR(error_code, error_msg) \
    raise_error(error_code, error_msg, __FILE__, __LINE__, __func__)

/**
 * \brief Wrapper for traceback function.
 * 
 * \param function_call Function call to be traced.
 * 
 * \return ErrorStatus struct.
 */
#define WRAP_TRACEBACK(function_call) \
    traceback(function_call, #function_call, __FILE__, __LINE__, __func__)

/**
 * \brief Make a error status struct with return code set to SUCCESS.
 * 
 * \return ErrorStatus struct with return code set to SUCCESS.
 */
ErrorStatus make_success_error_status(void);

/**
 * \brief Raise a warning and print to stderr.
 * 
 * \param warning_msg Warning message.
 * \param warning_file File where the warning occurs.
 * \param warning_line Line number where the warning occurs.
 * \param warning_func Function where the warning occurs.
 */
void raise_warning(
    const char *__restrict warning_msg,
    const char *__restrict warning_file,
    const int warning_line,
    const char *__restrict warning_func
);

/**
 * \brief Raise an error.
 * 
 * \param error_code Error code.
 * \param error_msg Error message.
 * \param error_file File where the error occurs.
 * \param error_line Line number where the error occurs.
 * \param error_func Function where the error occurs.
 * 
 * \return ErrorStatus struct.
 */
ErrorStatus raise_error(
    const int error_code,
    const char *__restrict error_msg,
    const char *__restrict error_file,
    const int error_line,
    const char *__restrict error_func
);

/**
 * \brief Stack traceback if error occurs.
 * 
 * \param error_status Pointer to the error status struct.
 * \param function_call_source_code Source code of the function call.
 * \param error_file File where the error occurs.
 * \param error_line Line number where the error occurs.
 * \param error_func Function where the error occurs.
 * 
 * \return ErrorStatus struct.
 */
ErrorStatus traceback(
    ErrorStatus error_status,
    const char *__restrict function_call_source_code,
    const char *__restrict error_file,
    const int error_line,
    const char *__restrict error_func
);

/**
 * \brief Free the memory allocated for the traceback string.
 */
void free_traceback(ErrorStatus *__restrict error_status);

/**
 * \brief Print the traceback string to stderr and free the memory.
 * 
 * \param error_status Pointer to the error status struct.
 */
void print_and_free_traceback(ErrorStatus *__restrict error_status);
 
#endif
