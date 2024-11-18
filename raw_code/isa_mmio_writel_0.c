static void isa_mmio_writel(void *opaque, target_phys_addr_t addr,

                               uint32_t val)

{

    cpu_outl(addr & IOPORTS_MASK, val);

}
