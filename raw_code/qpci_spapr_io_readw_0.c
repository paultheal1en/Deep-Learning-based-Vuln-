static uint16_t qpci_spapr_io_readw(QPCIBus *bus, void *addr)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    uint16_t v;

    if (port < s->pio.size) {

        v = readw(s->pio_cpu_base + port);

    } else {

        v = readw(s->mmio_cpu_base + port);

    }

    return bswap16(v);

}
