static void omap_lpg_write(void *opaque, target_phys_addr_t addr,

                           uint64_t value, unsigned size)

{

    struct omap_lpg_s *s = (struct omap_lpg_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 1) {

        return omap_badwidth_write8(opaque, addr, value);

    }



    switch (offset) {

    case 0x00:	/* LCR */

        if (~value & (1 << 6))					/* LPGRES */

            omap_lpg_reset(s);

        s->control = value & 0xff;

        omap_lpg_update(s);

        return;



    case 0x04:	/* PMR */

        s->power = value & 0x01;

        omap_lpg_update(s);

        return;



    default:

        OMAP_BAD_REG(addr);

        return;

    }

}
