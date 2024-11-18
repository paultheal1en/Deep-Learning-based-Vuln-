void qpci_memwrite(QPCIDevice *dev, void *data, const void *buf, size_t len)

{

    uintptr_t addr = (uintptr_t)data;



    g_assert(addr >= QPCI_PIO_LIMIT);

    dev->bus->memwrite(dev->bus, addr, buf, len);

}
