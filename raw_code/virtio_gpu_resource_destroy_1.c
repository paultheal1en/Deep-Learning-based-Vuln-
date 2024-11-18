static void virtio_gpu_resource_destroy(VirtIOGPU *g,

                                        struct virtio_gpu_simple_resource *res)

{

    pixman_image_unref(res->image);


    QTAILQ_REMOVE(&g->reslist, res, next);

    g->hostmem -= res->hostmem;

    g_free(res);

}