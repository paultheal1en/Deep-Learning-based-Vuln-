static void virtio_net_vmstate_change(void *opaque, int running, int reason)

{

    VirtIONet *n = opaque;

    if (!running) {

        return;

    }

    /* This is called when vm is started, it will start vhost backend if

     * appropriate e.g. after migration. */

    virtio_net_set_status(&n->vdev, n->vdev.status);

}
