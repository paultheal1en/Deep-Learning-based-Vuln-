static void omap_mcbsp_write(void *opaque, target_phys_addr_t addr,

                             uint64_t value, unsigned size)

{

    switch (size) {

    case 2: return omap_mcbsp_writeh(opaque, addr, value);

    case 4: return omap_mcbsp_writew(opaque, addr, value);

    default: return omap_badwidth_write16(opaque, addr, value);

    }

}
