static void pci_apb_iowritel (void *opaque, target_phys_addr_t addr,

                                uint32_t val)

{

    cpu_outl(addr & IOPORTS_MASK, bswap32(val));

}
