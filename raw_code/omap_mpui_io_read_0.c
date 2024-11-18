static uint64_t omap_mpui_io_read(void *opaque, target_phys_addr_t addr,

                                  unsigned size)

{

    if (size != 2) {

        return omap_badwidth_read16(opaque, addr);

    }



    if (addr == OMAP_MPUI_BASE)	/* CMR */

        return 0xfe4d;



    OMAP_BAD_REG(addr);

    return 0;

}
