vubr_backend_recv_cb(int sock, void *ctx)

{

    VubrDev *vubr = (VubrDev *) ctx;

    VuDev *dev = &vubr->vudev;

    VuVirtq *vq = vu_get_queue(dev, 0);

    VuVirtqElement *elem = NULL;

    struct iovec mhdr_sg[VIRTQUEUE_MAX_SIZE];

    struct virtio_net_hdr_mrg_rxbuf mhdr;

    unsigned mhdr_cnt = 0;

    int hdrlen = vubr->hdrlen;

    int i = 0;

    struct virtio_net_hdr hdr = {

        .flags = 0,

        .gso_type = VIRTIO_NET_HDR_GSO_NONE

    };



    DPRINT("\n\n   ***   IN UDP RECEIVE CALLBACK    ***\n\n");

    DPRINT("    hdrlen = %d\n", hdrlen);



    if (!vu_queue_enabled(dev, vq) ||

        !vu_queue_started(dev, vq) ||

        !vu_queue_avail_bytes(dev, vq, hdrlen, 0)) {

        DPRINT("Got UDP packet, but no available descriptors on RX virtq.\n");

        return;

    }



    do {

        struct iovec *sg;

        ssize_t ret, total = 0;

        unsigned int num;



        elem = vu_queue_pop(dev, vq, sizeof(VuVirtqElement));

        if (!elem) {

            break;

        }



        if (elem->in_num < 1) {

            fprintf(stderr, "virtio-net contains no in buffers\n");

            break;

        }



        sg = elem->in_sg;

        num = elem->in_num;

        if (i == 0) {

            if (hdrlen == 12) {

                mhdr_cnt = iov_copy(mhdr_sg, ARRAY_SIZE(mhdr_sg),

                                    sg, elem->in_num,

                                    offsetof(typeof(mhdr), num_buffers),

                                    sizeof(mhdr.num_buffers));

            }

            iov_from_buf(sg, elem->in_num, 0, &hdr, sizeof hdr);

            total += hdrlen;

            ret = iov_discard_front(&sg, &num, hdrlen);

            assert(ret == hdrlen);

        }



        struct msghdr msg = {

            .msg_name = (struct sockaddr *) &vubr->backend_udp_dest,

            .msg_namelen = sizeof(struct sockaddr_in),

            .msg_iov = sg,

            .msg_iovlen = elem->in_num,

            .msg_flags = MSG_DONTWAIT,

        };

        do {

            ret = recvmsg(vubr->backend_udp_sock, &msg, 0);

        } while (ret == -1 && (errno == EINTR));



        if (i == 0) {

            iov_restore_front(elem->in_sg, sg, hdrlen);

        }



        if (ret == -1) {

            if (errno == EWOULDBLOCK) {

                vu_queue_rewind(dev, vq, 1);

                break;

            }



            vubr_die("recvmsg()");

        }



        total += ret;

        iov_truncate(elem->in_sg, elem->in_num, total);

        vu_queue_fill(dev, vq, elem, total, i++);



        free(elem);

        elem = NULL;

    } while (false); /* could loop if DONTWAIT worked? */



    if (mhdr_cnt) {

        mhdr.num_buffers = i;

        iov_from_buf(mhdr_sg, mhdr_cnt,

                     0,

                     &mhdr.num_buffers, sizeof mhdr.num_buffers);

    }



    vu_queue_flush(dev, vq, i);

    vu_queue_notify(dev, vq);



    free(elem);

}
