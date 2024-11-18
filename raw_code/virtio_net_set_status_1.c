static void virtio_net_set_status(struct VirtIODevice *vdev, uint8_t status)

{

    VirtIONet *n = VIRTIO_NET(vdev);

    VirtIONetQueue *q;

    int i;

    uint8_t queue_status;



    virtio_net_vnet_endian_status(n, status);

    virtio_net_vhost_status(n, status);



    for (i = 0; i < n->max_queues; i++) {

        NetClientState *ncs = qemu_get_subqueue(n->nic, i);

        bool queue_started;

        q = &n->vqs[i];



        if ((!n->multiqueue && i != 0) || i >= n->curr_queues) {

            queue_status = 0;

        } else {

            queue_status = status;

        }

        queue_started =

            virtio_net_started(n, queue_status) && !n->vhost_started;



        if (queue_started) {

            qemu_flush_queued_packets(ncs);

        }



        if (!q->tx_waiting) {

            continue;

        }



        if (queue_started) {

            if (q->tx_timer) {

                timer_mod(q->tx_timer,

                               qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + n->tx_timeout);

            } else {

                qemu_bh_schedule(q->tx_bh);

            }

        } else {

            if (q->tx_timer) {

                timer_del(q->tx_timer);

            } else {

                qemu_bh_cancel(q->tx_bh);

            }

            if ((n->status & VIRTIO_NET_S_LINK_UP) == 0 &&

                (queue_status & VIRTIO_CONFIG_S_DRIVER_OK)) {

                /* if tx is waiting we are likely have some packets in tx queue

                 * and disabled notification */

                q->tx_waiting = 0;

                virtio_queue_set_notification(q->tx_vq, 1);

                virtio_net_drop_tx_queue_data(vdev, q->tx_vq);

            }

        }

    }

}
