static QVirtioPCIDevice *virtio_blk_pci_init(QPCIBus *bus, int slot)

{

    QVirtioPCIDevice *dev;



    dev = qvirtio_pci_device_find(bus, VIRTIO_ID_BLOCK);

    g_assert(dev != NULL);

    g_assert_cmphex(dev->vdev.device_type, ==, VIRTIO_ID_BLOCK);

    g_assert_cmphex(dev->pdev->devfn, ==, ((slot << 3) | PCI_FN));



    qvirtio_pci_device_enable(dev);

    qvirtio_reset(&dev->vdev);

    qvirtio_set_acknowledge(&dev->vdev);

    qvirtio_set_driver(&dev->vdev);



    return dev;

}
