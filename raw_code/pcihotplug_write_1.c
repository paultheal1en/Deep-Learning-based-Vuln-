static void pcihotplug_write(void *opaque, uint32_t addr, uint32_t val)

{

    struct pci_status *g = opaque;

    switch (addr) {

        case PCI_BASE:

            g->up = val;

            break;

        case PCI_BASE + 4:

            g->down = val;

            break;

   }



    PIIX4_DPRINTF("pcihotplug write %x <== %d\n", addr, val);

}
