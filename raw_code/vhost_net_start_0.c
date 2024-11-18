int vhost_net_start(struct vhost_net *net,

                    VirtIODevice *dev)

{

    struct vhost_vring_file file = { };

    int r;

    if (net->dev.acked_features & (1 << VIRTIO_NET_F_MRG_RXBUF)) {

        tap_set_vnet_hdr_len(net->vc,

                             sizeof(struct virtio_net_hdr_mrg_rxbuf));

    }



    net->dev.nvqs = 2;

    net->dev.vqs = net->vqs;

    r = vhost_dev_start(&net->dev, dev);

    if (r < 0) {

        return r;

    }



    net->vc->info->poll(net->vc, false);

    qemu_set_fd_handler(net->backend, NULL, NULL, NULL);

    file.fd = net->backend;

    for (file.index = 0; file.index < net->dev.nvqs; ++file.index) {

        r = ioctl(net->dev.control, VHOST_NET_SET_BACKEND, &file);

        if (r < 0) {

            r = -errno;

            goto fail;

        }

    }

    return 0;

fail:

    file.fd = -1;

    while (--file.index >= 0) {

        int r = ioctl(net->dev.control, VHOST_NET_SET_BACKEND, &file);

        assert(r >= 0);

    }

    net->vc->info->poll(net->vc, true);

    vhost_dev_stop(&net->dev, dev);

    if (net->dev.acked_features & (1 << VIRTIO_NET_F_MRG_RXBUF)) {

        tap_set_vnet_hdr_len(net->vc, sizeof(struct virtio_net_hdr));

    }

    return r;

}
