void av_opencl_uninit(void)

{

    cl_int status;

    int i;

    LOCK_OPENCL

    gpu_env.init_count--;

    if (gpu_env.is_user_created)

        goto end;

    if ((gpu_env.init_count > 0) || (gpu_env.kernel_count > 0))

        goto end;

    for (i = 0; i < gpu_env.program_count; i++) {

        if (gpu_env.programs[i]) {

            status = clReleaseProgram(gpu_env.programs[i]);

            if (status != CL_SUCCESS) {

                av_log(&openclutils, AV_LOG_ERROR, "Could not release OpenCL program: %s\n", opencl_errstr(status));

            }

            gpu_env.programs[i] = NULL;

        }

    }

    if (gpu_env.command_queue) {

        status = clReleaseCommandQueue(gpu_env.command_queue);

        if (status != CL_SUCCESS) {

            av_log(&openclutils, AV_LOG_ERROR, "Could not release OpenCL command queue: %s\n", opencl_errstr(status));

        }

        gpu_env.command_queue = NULL;

    }

    if (gpu_env.context) {

        status = clReleaseContext(gpu_env.context);

        if (status != CL_SUCCESS) {

            av_log(&openclutils, AV_LOG_ERROR, "Could not release OpenCL context: %s\n", opencl_errstr(status));

        }

        gpu_env.context = NULL;

    }

    av_freep(&(gpu_env.device_ids));

end:

    UNLOCK_OPENCL

}
