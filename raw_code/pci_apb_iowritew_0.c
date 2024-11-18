static void pci_apb_iowritew (void *opaque, target_phys_addr_t addr,

                                  uint32_t val)

{

    cpu_outw(addr & IOPORTS_MASK, bswap16(val));

}
