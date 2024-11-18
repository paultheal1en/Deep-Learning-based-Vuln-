struct vhost_net *vhost_net_init(NetClientState *backend, int devfd,

                                 bool force)

{

    int r;

    struct vhost_net *net = g_malloc(sizeof *net);

    if (!backend) {

        fprintf(stderr, "vhost-net requires backend to be setup\n");

        goto fail;

    }

    r = vhost_net_get_fd(backend);

    if (r < 0) {

        goto fail;

    }

    net->nc = backend;

    net->dev.backend_features = tap_has_vnet_hdr(backend) ? 0 :

        (1 << VHOST_NET_F_VIRTIO_NET_HDR);

    net->backend = r;



    net->dev.nvqs = 2;

    net->dev.vqs = net->vqs;



    r = vhost_dev_init(&net->dev, devfd, "/dev/vhost-net", force);

    if (r < 0) {

        goto fail;

    }

    if (!tap_has_vnet_hdr_len(backend,

                              sizeof(struct virtio_net_hdr_mrg_rxbuf))) {

        net->dev.features &= ~(1 << VIRTIO_NET_F_MRG_RXBUF);

    }

    if (~net->dev.features & net->dev.backend_features) {

        fprintf(stderr, "vhost lacks feature mask %" PRIu64 " for backend\n",

                (uint64_t)(~net->dev.features & net->dev.backend_features));

        vhost_dev_cleanup(&net->dev);

        goto fail;

    }



    /* Set sane init value. Override when guest acks. */

    vhost_net_ack_features(net, 0);

    return net;

fail:

    g_free(net);

    return NULL;

}
