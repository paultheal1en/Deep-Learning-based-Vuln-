uint64_t qpci_io_readq(QPCIDevice *dev, void *data)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        return dev->bus->pio_readq(dev->bus, addr);

    } else {

        uint64_t val;

        dev->bus->memread(dev->bus, addr, &val, sizeof(val));

        return le64_to_cpu(val);

    }

}
