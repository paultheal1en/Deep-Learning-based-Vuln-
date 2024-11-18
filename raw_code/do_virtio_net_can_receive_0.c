static int do_virtio_net_can_receive(VirtIONet *n, int bufsize)

{

    if (!virtio_queue_ready(n->rx_vq) ||

        !(n->vdev.status & VIRTIO_CONFIG_S_DRIVER_OK))

        return 0;



    if (virtio_queue_empty(n->rx_vq) ||

        (n->mergeable_rx_bufs &&

         !virtqueue_avail_bytes(n->rx_vq, bufsize, 0))) {

        virtio_queue_set_notification(n->rx_vq, 1);

        return 0;

    }



    virtio_queue_set_notification(n->rx_vq, 0);

    return 1;

}
