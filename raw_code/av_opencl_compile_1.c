cl_program av_opencl_compile(const char *program_name, const char *build_opts)

{

    int i;

    cl_int status, build_status;

    int kernel_code_idx = 0;

    const char *kernel_source;

    size_t kernel_code_len;

    char* ptr = NULL;

    cl_program program = NULL;

    size_t log_size;

    char *log = NULL;



    LOCK_OPENCL;

    for (i = 0; i < opencl_ctx.kernel_code_count; i++) {

        // identify a program using a unique name within the kernel source

        ptr = av_stristr(opencl_ctx.kernel_code[i].kernel_string, program_name);

        if (ptr && !opencl_ctx.kernel_code[i].is_compiled) {

            kernel_source = opencl_ctx.kernel_code[i].kernel_string;

            kernel_code_len = strlen(opencl_ctx.kernel_code[i].kernel_string);

            kernel_code_idx = i;

            break;

        }

    }

    if (!kernel_source) {

        av_log(&opencl_ctx, AV_LOG_ERROR,

               "Unable to find OpenCL kernel source '%s'\n", program_name);

        goto end;

    }



    /* create a CL program from kernel source */

    program = clCreateProgramWithSource(opencl_ctx.context, 1, &kernel_source, &kernel_code_len, &status);

    if(status != CL_SUCCESS) {

        av_log(&opencl_ctx, AV_LOG_ERROR,

               "Unable to create OpenCL program '%s': %s\n", program_name, av_opencl_errstr(status));

        program = NULL;

        goto end;

    }



    build_status = clBuildProgram(program, 1, &(opencl_ctx.device_id), build_opts, NULL, NULL);

    status = clGetProgramBuildInfo(program, opencl_ctx.device_id,

                                   CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    if (status != CL_SUCCESS) {

        av_log(&opencl_ctx, AV_LOG_WARNING,

               "Failed to get compilation log: %s\n",

               av_opencl_errstr(status));

    } else {

        log = av_malloc(log_size);

        if (log) {

            status = clGetProgramBuildInfo(program, opencl_ctx.device_id,

                                           CL_PROGRAM_BUILD_LOG, log_size,

                                           log, NULL);

            if (status != CL_SUCCESS) {

                av_log(&opencl_ctx, AV_LOG_WARNING,

                       "Failed to get compilation log: %s\n",

                       av_opencl_errstr(status));

            } else {

                int level = build_status == CL_SUCCESS ? AV_LOG_DEBUG :

                                                         AV_LOG_ERROR;

                av_log(&opencl_ctx, level, "Compilation log:\n%s\n", log);

            }

        }

        av_freep(&log);

    }

    if (build_status != CL_SUCCESS) {

        av_log(&opencl_ctx, AV_LOG_ERROR,

               "Compilation failed with OpenCL program '%s': %s\n",

               program_name, av_opencl_errstr(build_status));

        program = NULL;

        goto end;

    }



    opencl_ctx.kernel_code[kernel_code_idx].is_compiled = 1;

end:

    UNLOCK_OPENCL;

    return program;

}
