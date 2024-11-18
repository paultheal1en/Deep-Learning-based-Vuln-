static uint32_t isa_mmio_readl(void *opaque, target_phys_addr_t addr)

{

    return cpu_inl(addr & IOPORTS_MASK);

}
