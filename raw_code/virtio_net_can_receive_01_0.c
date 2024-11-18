static int virtio_net_can_receive(void *opaque)

{

    VirtIONet *n = opaque;



    return do_virtio_net_can_receive(n, VIRTIO_NET_MAX_BUFSIZE);

}
