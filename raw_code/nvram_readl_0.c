static uint32_t nvram_readl (void *opaque, target_phys_addr_t addr)

{

    M48t59State *NVRAM = opaque;

    uint32_t retval;



    retval = m48t59_read(NVRAM, addr) << 24;

    retval |= m48t59_read(NVRAM, addr + 1) << 16;

    retval |= m48t59_read(NVRAM, addr + 2) << 8;

    retval |= m48t59_read(NVRAM, addr + 3);

    return retval;

}
