static void virtio_pci_reset(void *opaque)

{

    VirtIOPCIProxy *proxy = opaque;

    virtio_reset(proxy->vdev);

    msix_reset(&proxy->pci_dev);

}
