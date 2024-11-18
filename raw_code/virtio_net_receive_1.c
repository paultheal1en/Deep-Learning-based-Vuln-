static ssize_t virtio_net_receive(NetClientState *nc, const uint8_t *buf, size_t size)

{

    VirtIONet *n = qemu_get_nic_opaque(nc);

    VirtIONetQueue *q = virtio_net_get_subqueue(nc);

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    struct iovec mhdr_sg[VIRTQUEUE_MAX_SIZE];

    struct virtio_net_hdr_mrg_rxbuf mhdr;

    unsigned mhdr_cnt = 0;

    size_t offset, i, guest_offset;



    if (!virtio_net_can_receive(nc)) {

        return -1;

    }



    /* hdr_len refers to the header we supply to the guest */

    if (!virtio_net_has_buffers(q, size + n->guest_hdr_len - n->host_hdr_len)) {

        return 0;

    }



    if (!receive_filter(n, buf, size))

        return size;



    offset = i = 0;



    while (offset < size) {

        VirtQueueElement elem;

        int len, total;

        const struct iovec *sg = elem.in_sg;



        total = 0;



        if (virtqueue_pop(q->rx_vq, &elem) == 0) {

            if (i == 0)

                return -1;

            error_report("virtio-net unexpected empty queue: "

                         "i %zd mergeable %d offset %zd, size %zd, "

                         "guest hdr len %zd, host hdr len %zd "

                         "guest features 0x%" PRIx64,

                         i, n->mergeable_rx_bufs, offset, size,

                         n->guest_hdr_len, n->host_hdr_len,

                         vdev->guest_features);

            exit(1);

        }



        if (elem.in_num < 1) {

            error_report("virtio-net receive queue contains no in buffers");

            exit(1);

        }



        if (i == 0) {

            assert(offset == 0);

            if (n->mergeable_rx_bufs) {

                mhdr_cnt = iov_copy(mhdr_sg, ARRAY_SIZE(mhdr_sg),

                                    sg, elem.in_num,

                                    offsetof(typeof(mhdr), num_buffers),

                                    sizeof(mhdr.num_buffers));

            }



            receive_header(n, sg, elem.in_num, buf, size);

            offset = n->host_hdr_len;

            total += n->guest_hdr_len;

            guest_offset = n->guest_hdr_len;

        } else {

            guest_offset = 0;

        }



        /* copy in packet.  ugh */

        len = iov_from_buf(sg, elem.in_num, guest_offset,

                           buf + offset, size - offset);

        total += len;

        offset += len;

        /* If buffers can't be merged, at this point we

         * must have consumed the complete packet.

         * Otherwise, drop it. */

        if (!n->mergeable_rx_bufs && offset < size) {

#if 0

            error_report("virtio-net truncated non-mergeable packet: "

                         "i %zd mergeable %d offset %zd, size %zd, "

                         "guest hdr len %zd, host hdr len %zd",

                         i, n->mergeable_rx_bufs,

                         offset, size, n->guest_hdr_len, n->host_hdr_len);

#endif

            return size;

        }



        /* signal other side */

        virtqueue_fill(q->rx_vq, &elem, total, i++);

    }



    if (mhdr_cnt) {

        virtio_stw_p(vdev, &mhdr.num_buffers, i);

        iov_from_buf(mhdr_sg, mhdr_cnt,

                     0,

                     &mhdr.num_buffers, sizeof mhdr.num_buffers);

    }



    virtqueue_flush(q->rx_vq, i);

    virtio_notify(vdev, q->rx_vq);



    return size;

}
