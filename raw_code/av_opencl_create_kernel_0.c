int av_opencl_create_kernel(AVOpenCLKernelEnv *env, const char *kernel_name)

{

    cl_int status;

    int i, ret = 0;

    LOCK_OPENCL;

    if (strlen(kernel_name) + 1 > AV_OPENCL_MAX_KERNEL_NAME_SIZE) {

        av_log(&openclutils, AV_LOG_ERROR, "Created kernel name %s is too long\n", kernel_name);

        ret = AVERROR(EINVAL);

        goto end;

    }

    if (!env->kernel) {

        if (gpu_env.kernel_count >= MAX_KERNEL_NUM) {

            av_log(&openclutils, AV_LOG_ERROR,

            "Could not create kernel with name '%s', maximum number of kernels %d already reached\n",

                kernel_name, MAX_KERNEL_NUM);

            ret = AVERROR(EINVAL);

            goto end;

        }

        for (i = 0; i < gpu_env.program_count; i++) {

            env->kernel = clCreateKernel(gpu_env.programs[i], kernel_name, &status);

            if (status == CL_SUCCESS)

                break;

        }

        if (status != CL_SUCCESS) {

            av_log(&openclutils, AV_LOG_ERROR, "Could not create OpenCL kernel: %s\n", opencl_errstr(status));

            ret = AVERROR_EXTERNAL;

            goto end;

        }

        gpu_env.kernel_count++;

        env->command_queue = gpu_env.command_queue;

        av_strlcpy(env->kernel_name, kernel_name, sizeof(env->kernel_name));

    }

end:

    UNLOCK_OPENCL;

    return ret;

}
