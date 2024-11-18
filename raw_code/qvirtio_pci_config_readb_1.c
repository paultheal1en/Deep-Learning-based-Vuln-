static uint8_t qvirtio_pci_config_readb(QVirtioDevice *d, uint64_t off)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    return qpci_io_readb(dev->pdev, CONFIG_BASE(dev) + off);

}
