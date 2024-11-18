static unsigned virtio_pci_get_features(void *opaque)

{

    unsigned ret = 0;

    ret |= (1 << VIRTIO_F_NOTIFY_ON_EMPTY);

    ret |= (1 << VIRTIO_RING_F_INDIRECT_DESC);

    ret |= (1 << VIRTIO_F_BAD_FEATURE);

    return ret;

}
