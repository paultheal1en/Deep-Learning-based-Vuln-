static int virtio_9p_init_pci(PCIDevice *pci_dev)

{

    VirtIOPCIProxy *proxy = DO_UPCAST(VirtIOPCIProxy, pci_dev, pci_dev);

    VirtIODevice *vdev;



    vdev = virtio_9p_init(&pci_dev->qdev, &proxy->fsconf);

    vdev->nvectors = proxy->nvectors;

    virtio_init_pci(proxy, vdev,

                    PCI_VENDOR_ID_REDHAT_QUMRANET,

                    0x1009,

                    0x2,

                    0x00);

    /* make the actual value visible */

    proxy->nvectors = vdev->nvectors;

    return 0;

}
