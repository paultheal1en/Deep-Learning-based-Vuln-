static uint64_t omap_sti_fifo_read(void *opaque, target_phys_addr_t addr,

                                   unsigned size)

{

    OMAP_BAD_REG(addr);

    return 0;

}
