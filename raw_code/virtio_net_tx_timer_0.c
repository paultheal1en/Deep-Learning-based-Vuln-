static void virtio_net_tx_timer(void *opaque)

{

    VirtIONetQueue *q = opaque;

    VirtIONet *n = q->n;

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    assert(vdev->vm_running);



    q->tx_waiting = 0;



    /* Just in case the driver is not ready on more */

    if (!(vdev->status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        return;

    }



    virtio_queue_set_notification(q->tx_vq, 1);

    virtio_net_flush_tx(q);

}
