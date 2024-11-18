void qvirtio_pci_device_disable(QVirtioPCIDevice *d)

{

    qpci_iounmap(d->pdev, d->addr);

    d->addr = NULL;

}
