int av_opencl_buffer_read_image(uint8_t **dst_data, int *plane_size, int plane_num,

                                       cl_mem src_cl_buf, size_t cl_buffer_size)

{

    int i,buffer_size = 0,ret = 0;

    uint8_t *temp;

    void *mapped;

    cl_int status;

    if ((unsigned int)plane_num > 8) {

        return AVERROR(EINVAL);

    }

    for (i = 0;i < plane_num;i++) {

        buffer_size += plane_size[i];

    }

    if (buffer_size > cl_buffer_size) {

        av_log(&openclutils, AV_LOG_ERROR, "Cannot write image to CPU buffer: OpenCL buffer too small\n");

        return AVERROR(EINVAL);

    }

    mapped = clEnqueueMapBuffer(gpu_env.command_queue, src_cl_buf,

                                      CL_TRUE,CL_MAP_READ, 0, buffer_size,

                                      0, NULL, NULL, &status);



    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not map OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    temp = mapped;

    if (ret >= 0) {

        for (i = 0;i < plane_num;i++) {

            memcpy(dst_data[i], temp, plane_size[i]);

            temp += plane_size[i];

        }

    }

    status = clEnqueueUnmapMemObject(gpu_env.command_queue, src_cl_buf, mapped, 0, NULL, NULL);

    if (status != CL_SUCCESS) {

        av_log(&openclutils, AV_LOG_ERROR, "Could not unmap OpenCL buffer: %s\n", opencl_errstr(status));

        return AVERROR_EXTERNAL;

    }

    return 0;

}
