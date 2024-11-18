static uint8_t qpci_spapr_io_readb(QPCIBus *bus, void *addr)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    uint8_t v;

    if (port < s->pio.size) {

        v = readb(s->pio_cpu_base + port);

    } else {

        v = readb(s->mmio_cpu_base + port);

    }

    return v;

}
