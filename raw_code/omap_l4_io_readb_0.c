static uint32_t omap_l4_io_readb(void *opaque, target_phys_addr_t addr)

{

    unsigned int i = (addr - OMAP2_L4_BASE) >> TARGET_PAGE_BITS;



    return omap_l4_io_readb_fn[i](omap_l4_io_opaque[i], addr);

}
