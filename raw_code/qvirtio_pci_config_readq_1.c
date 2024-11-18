static uint64_t qvirtio_pci_config_readq(QVirtioDevice *d, uint64_t off)

{

    QVirtioPCIDevice *dev = (QVirtioPCIDevice *)d;

    uint64_t val;



    val = qpci_io_readq(dev->pdev, CONFIG_BASE(dev) + off);

    if (qvirtio_is_big_endian(d)) {

        val = bswap64(val);

    }



    return val;

}
