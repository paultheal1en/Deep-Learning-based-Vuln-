static uint32_t isa_mmio_readb (void *opaque, target_phys_addr_t addr)

{

    return cpu_inb(addr & IOPORTS_MASK);

}
