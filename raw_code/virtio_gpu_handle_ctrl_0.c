static void virtio_gpu_handle_ctrl(VirtIODevice *vdev, VirtQueue *vq)

{

    VirtIOGPU *g = VIRTIO_GPU(vdev);

    struct virtio_gpu_ctrl_command *cmd;



    if (!virtio_queue_ready(vq)) {

        return;

    }



#ifdef CONFIG_VIRGL

    if (!g->renderer_inited && g->use_virgl_renderer) {

        virtio_gpu_virgl_init(g);

        g->renderer_inited = true;

    }

#endif



    cmd = g_new(struct virtio_gpu_ctrl_command, 1);

    while (virtqueue_pop(vq, &cmd->elem)) {

        cmd->vq = vq;

        cmd->error = 0;

        cmd->finished = false;

        cmd->waiting = false;

        QTAILQ_INSERT_TAIL(&g->cmdq, cmd, next);

        cmd = g_new(struct virtio_gpu_ctrl_command, 1);

    }

    g_free(cmd);



    virtio_gpu_process_cmdq(g);



#ifdef CONFIG_VIRGL

    if (g->use_virgl_renderer) {

        virtio_gpu_virgl_fence_poll(g);

    }

#endif

}
