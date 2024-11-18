void qvirtio_pci_device_enable(QVirtioPCIDevice *d)

{

    qpci_device_enable(d->pdev);

    d->addr = qpci_iomap(d->pdev, 0, NULL);

    g_assert(d->addr != NULL);

}
