static uint32_t qvirtio_pci_config_readl(QVirtioDevice *d, uint64_t off)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    uint32_t value;



    value = qpci_io_readl(dev->pdev, CONFIG_BASE(dev) + off);

    if (qvirtio_is_big_endian(d)) {

        value = bswap32(value);

    }

    return value;

}
