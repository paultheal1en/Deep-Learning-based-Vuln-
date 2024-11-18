static uint16_t qvirtio_pci_get_queue_size(QVirtioDevice *d)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    return qpci_io_readw(dev->pdev, dev->addr + VIRTIO_PCI_QUEUE_NUM);

}
