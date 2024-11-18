static void isa_mmio_writew(void *opaque, target_phys_addr_t addr,

                               uint32_t val)

{

    cpu_outw(addr & IOPORTS_MASK, val);

}
