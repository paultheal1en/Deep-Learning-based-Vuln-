static void qpci_spapr_io_writew(QPCIBus *bus, void *addr, uint16_t value)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    value = bswap16(value);

    if (port < s->pio.size) {

        writew(s->pio_cpu_base + port, value);

    } else {

        writew(s->mmio_cpu_base + port, value);

    }

}
