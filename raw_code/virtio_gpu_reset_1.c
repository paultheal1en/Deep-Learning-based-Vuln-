static void virtio_gpu_reset(VirtIODevice *vdev)

{

    VirtIOGPU *g = VIRTIO_GPU(vdev);

    struct virtio_gpu_simple_resource *res, *tmp;

    int i;



    g->enable = 0;



    QTAILQ_FOREACH_SAFE(res, &g->reslist, next, tmp) {

        virtio_gpu_resource_destroy(g, res);

    }

    for (i = 0; i < g->conf.max_outputs; i++) {

#if 0

        g->req_state[i].x = 0;

        g->req_state[i].y = 0;

        if (i == 0) {

            g->req_state[0].width = 1024;

            g->req_state[0].height = 768;

        } else {

            g->req_state[i].width = 0;

            g->req_state[i].height = 0;

        }

#endif

        g->scanout[i].resource_id = 0;

        g->scanout[i].width = 0;

        g->scanout[i].height = 0;

        g->scanout[i].x = 0;

        g->scanout[i].y = 0;

        g->scanout[i].ds = NULL;

    }

    g->enabled_output_bitmask = 1;



#ifdef CONFIG_VIRGL

    if (g->use_virgl_renderer) {

        virtio_gpu_virgl_reset(g);

        g->use_virgl_renderer = 0;

    }

#endif

}
