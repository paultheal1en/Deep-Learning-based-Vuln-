static int virtio_net_handle_offloads(VirtIONet *n, uint8_t cmd,

                                     struct iovec *iov, unsigned int iov_cnt)

{

    VirtIODevice *vdev = VIRTIO_DEVICE(n);

    uint64_t offloads;

    size_t s;



    if (!((1 << VIRTIO_NET_F_CTRL_GUEST_OFFLOADS) & vdev->guest_features)) {

        return VIRTIO_NET_ERR;

    }



    s = iov_to_buf(iov, iov_cnt, 0, &offloads, sizeof(offloads));

    if (s != sizeof(offloads)) {

        return VIRTIO_NET_ERR;

    }



    if (cmd == VIRTIO_NET_CTRL_GUEST_OFFLOADS_SET) {

        uint64_t supported_offloads;



        if (!n->has_vnet_hdr) {

            return VIRTIO_NET_ERR;

        }



        supported_offloads = virtio_net_supported_guest_offloads(n);

        if (offloads & ~supported_offloads) {

            return VIRTIO_NET_ERR;

        }



        n->curr_guest_offloads = offloads;

        virtio_net_apply_guest_offloads(n);



        return VIRTIO_NET_OK;

    } else {

        return VIRTIO_NET_ERR;

    }

}
