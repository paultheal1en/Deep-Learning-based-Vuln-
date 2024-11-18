static void virtio_net_set_multiqueue(VirtIONet *n, int multiqueue, int ctrl)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    int i, max = multiqueue ? n->max_queues : 1;



    n->multiqueue = multiqueue;



    for (i = 2; i <= n->max_queues * 2 + 1; i++) {

        virtio_del_queue(vdev, i);

    }



    for (i = 1; i < max; i++) {

        n->vqs[i].rx_vq = virtio_add_queue(vdev, 256, virtio_net_handle_rx);

        if (n->vqs[i].tx_timer) {

            n->vqs[i].tx_vq =

                virtio_add_queue(vdev, 256, virtio_net_handle_tx_timer);

            n->vqs[i].tx_timer = qemu_new_timer_ns(vm_clock,

                                                   virtio_net_tx_timer,

                                                   &n->vqs[i]);

        } else {

            n->vqs[i].tx_vq =

                virtio_add_queue(vdev, 256, virtio_net_handle_tx_bh);

            n->vqs[i].tx_bh = qemu_bh_new(virtio_net_tx_bh, &n->vqs[i]);

        }



        n->vqs[i].tx_waiting = 0;

        n->vqs[i].n = n;

    }



    if (ctrl) {

        n->ctrl_vq = virtio_add_queue(vdev, 64, virtio_net_handle_ctrl);

    }



    virtio_net_set_queues(n);

}
