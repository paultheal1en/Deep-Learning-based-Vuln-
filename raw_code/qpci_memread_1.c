void qpci_memread(QPCIDevice *dev, void *data, void *buf, size_t len)

{

    uintptr_t addr = (uintptr_t)data;



    g_assert(addr >= QPCI_PIO_LIMIT);

    dev->bus->memread(dev->bus, addr, buf, len);

}
