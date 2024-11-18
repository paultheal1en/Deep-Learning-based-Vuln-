int av_opencl_buffer_write_image(cl_mem dst_cl_buf, size_t cl_buffer_size, int dst_cl_offset,

                                                    uint8_t **src_data, int *plane_size, int plane_num)

{

    int i, buffer_size = 0;

    uint8_t *temp;

    cl_int status;

    void *mapped;

    if ((unsigned int)plane_num > 8) {

        return AVERROR(EINVAL);

    }

    for (i = 0;i < plane_num;i++) {

        buffer_size += plane_size[i];

    }

    if (buffer_size > cl_buffer_size) {

        av_log(&openclutils, AV_LOG_ERROR, "Cannot write image to OpenCL buffer: buffer too small\n");

        return AVERROR(EINVAL);

    }

    mapped = clEnqueueMapBuffer(gpu_env.command_queue, dst_cl_buf,

                                      CL_TRUE,CL_MAP_WRITE, 0, buffer_size + dst_cl_offset,

                                      0, NULL, NULL, &status);

    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not map OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    temp = mapped;

    temp += dst_cl_offset;

    for (i = 0; i < plane_num; i++) {

        memcpy(temp, src_data[i], plane_size[i]);

        temp += plane_size[i];

    }

    status = clEnqueueUnmapMemObject(gpu_env.command_queue, dst_cl_buf, mapped, 0, NULL, NULL);

    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not unmap OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    return 0;

}
