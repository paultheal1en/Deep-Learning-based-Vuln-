static uint32_t pci_apb_ioreadb (void *opaque, target_phys_addr_t addr)

{

    uint32_t val;



    val = cpu_inb(addr & IOPORTS_MASK);

    return val;

}
