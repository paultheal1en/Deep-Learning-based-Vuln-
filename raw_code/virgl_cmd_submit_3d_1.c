static void virgl_cmd_submit_3d(VirtIOGPU *g,

                                struct virtio_gpu_ctrl_command *cmd)

{

    struct virtio_gpu_cmd_submit cs;

    void *buf;

    size_t s;



    VIRTIO_GPU_FILL_CMD(cs);

    trace_virtio_gpu_cmd_ctx_submit(cs.hdr.ctx_id, cs.size);



    buf = g_malloc(cs.size);

    s = iov_to_buf(cmd->elem.out_sg, cmd->elem.out_num,

                   sizeof(cs), buf, cs.size);

    if (s != cs.size) {

        qemu_log_mask(LOG_GUEST_ERROR, "%s: size mismatch (%zd/%d)",

                      __func__, s, cs.size);

        cmd->error = VIRTIO_GPU_RESP_ERR_INVALID_PARAMETER;

        return;

    }



    if (virtio_gpu_stats_enabled(g->conf)) {

        g->stats.req_3d++;

        g->stats.bytes_3d += cs.size;

    }



    virgl_renderer_submit_cmd(buf, cs.hdr.ctx_id, cs.size / 4);



    g_free(buf);

}
