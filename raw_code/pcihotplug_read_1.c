static uint32_t pcihotplug_read(void *opaque, uint32_t addr)

{

    uint32_t val = 0;

    struct pci_status *g = opaque;

    switch (addr) {

        case PCI_BASE:

            val = g->up;

            break;

        case PCI_BASE + 4:

            val = g->down;

            break;

        default:

            break;

    }



    PIIX4_DPRINTF("pcihotplug read %x == %x\n", addr, val);

    return val;

}
