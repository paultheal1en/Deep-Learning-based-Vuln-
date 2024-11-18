static void qvirtio_pci_set_queue_address(QVirtioDevice *d, uint32_t pfn)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    qpci_io_writel(dev->pdev, dev->addr + VIRTIO_PCI_QUEUE_PFN, pfn);

}
