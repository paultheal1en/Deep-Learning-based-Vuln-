static void qpci_spapr_io_writel(QPCIBus *bus, void *addr, uint32_t value)

{

    QPCIBusSPAPR *s = container_of(bus, QPCIBusSPAPR, bus);

    uint64_t port = (uintptr_t)addr;

    value = bswap32(value);

    if (port < s->pio.size) {

        writel(s->pio_cpu_base + port, value);

    } else {

        writel(s->mmio_cpu_base + port, value);

    }

}
