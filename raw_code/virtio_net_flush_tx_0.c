static int32_t virtio_net_flush_tx(VirtIONetQueue *q)

{

    VirtIONet *n = q->n;

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    VirtQueueElement elem;

    int32_t num_packets = 0;

    int queue_index = vq2q(virtio_get_queue_index(q->tx_vq));

    if (!(vdev->status & VIRTIO_CONFIG_S_DRIVER_OK)) {

        return num_packets;

    }



    if (q->async_tx.elem.out_num) {

        virtio_queue_set_notification(q->tx_vq, 0);

        return num_packets;

    }



    while (virtqueue_pop(q->tx_vq, &elem)) {

        ssize_t ret;

        unsigned int out_num = elem.out_num;

        struct iovec *out_sg = &elem.out_sg[0];

        struct iovec sg[VIRTQUEUE_MAX_SIZE], sg2[VIRTQUEUE_MAX_SIZE + 1];

        struct virtio_net_hdr_mrg_rxbuf mhdr;



        if (out_num < 1) {

            error_report("virtio-net header not in first element");

            exit(1);

        }



        if (n->has_vnet_hdr) {

            if (iov_to_buf(out_sg, out_num, 0, &mhdr, n->guest_hdr_len) <

                n->guest_hdr_len) {

                error_report("virtio-net header incorrect");

                exit(1);

            }

            if (virtio_needs_swap(vdev)) {

                virtio_net_hdr_swap(vdev, (void *) &mhdr);

                sg2[0].iov_base = &mhdr;

                sg2[0].iov_len = n->guest_hdr_len;

                out_num = iov_copy(&sg2[1], ARRAY_SIZE(sg2) - 1,

                                   out_sg, out_num,

                                   n->guest_hdr_len, -1);

                if (out_num == VIRTQUEUE_MAX_SIZE) {

                    goto drop;

		}

                out_num += 1;

                out_sg = sg2;

	    }

        }

        /*

         * If host wants to see the guest header as is, we can

         * pass it on unchanged. Otherwise, copy just the parts

         * that host is interested in.

         */

        assert(n->host_hdr_len <= n->guest_hdr_len);

        if (n->host_hdr_len != n->guest_hdr_len) {

            unsigned sg_num = iov_copy(sg, ARRAY_SIZE(sg),

                                       out_sg, out_num,

                                       0, n->host_hdr_len);

            sg_num += iov_copy(sg + sg_num, ARRAY_SIZE(sg) - sg_num,

                             out_sg, out_num,

                             n->guest_hdr_len, -1);

            out_num = sg_num;

            out_sg = sg;

        }



        ret = qemu_sendv_packet_async(qemu_get_subqueue(n->nic, queue_index),

                                      out_sg, out_num, virtio_net_tx_complete);

        if (ret == 0) {

            virtio_queue_set_notification(q->tx_vq, 0);

            q->async_tx.elem = elem;

            return -EBUSY;

        }



drop:

        virtqueue_push(q->tx_vq, &elem, 0);

        virtio_notify(vdev, q->tx_vq);



        if (++num_packets >= n->tx_burst) {

            break;

        }

    }

    return num_packets;

}
