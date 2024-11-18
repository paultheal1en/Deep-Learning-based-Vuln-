int av_opencl_buffer_read(uint8_t *dst_buf, cl_mem src_cl_buf, size_t buf_size)

{

    cl_int status;

    void *mapped = clEnqueueMapBuffer(gpu_env.command_queue, src_cl_buf,

                                      CL_TRUE,CL_MAP_READ, 0, buf_size,

                                      0, NULL, NULL, &status);



    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not map OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    memcpy(dst_buf, mapped, buf_size);



    status = clEnqueueUnmapMemObject(gpu_env.command_queue, src_cl_buf, mapped, 0, NULL, NULL);

    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not unmap OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    return 0;

}
