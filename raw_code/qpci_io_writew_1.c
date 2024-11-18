void qpci_io_writew(QPCIDevice *dev, void *data, uint16_t value)

{

    uintptr_t addr = (uintptr_t)data;



    if (addr < QPCI_PIO_LIMIT) {

        dev->bus->pio_writew(dev->bus, addr, value);

    } else {

        value = cpu_to_le16(value);

        dev->bus->memwrite(dev->bus, addr, &value, sizeof(value));

    }

}
