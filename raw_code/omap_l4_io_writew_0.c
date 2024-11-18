static void omap_l4_io_writew(void *opaque, target_phys_addr_t addr,

                uint32_t value)

{

    unsigned int i = (addr - OMAP2_L4_BASE) >> TARGET_PAGE_BITS;



    return omap_l4_io_writew_fn[i](omap_l4_io_opaque[i], addr, value);

}
