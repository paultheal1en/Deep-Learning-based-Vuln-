static int virtio_net_can_receive(VLANClientState *vc)

{

    VirtIONet *n = vc->opaque;



    return do_virtio_net_can_receive(n, VIRTIO_NET_MAX_BUFSIZE);

}
