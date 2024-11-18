static uint32_t pci_up_read(void *opaque, uint32_t addr)

{

    PIIX4PMState *s = opaque;

    uint32_t val = s->pci0_status.up;



    PIIX4_DPRINTF("pci_up_read %x\n", val);

    return val;

}
