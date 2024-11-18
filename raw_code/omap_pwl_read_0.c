static uint64_t omap_pwl_read(void *opaque, target_phys_addr_t addr,

                              unsigned size)

{

    struct omap_pwl_s *s = (struct omap_pwl_s *) opaque;

    int offset = addr & OMAP_MPUI_REG_MASK;



    if (size != 1) {

        return omap_badwidth_read8(opaque, addr);

    }



    switch (offset) {

    case 0x00:	/* PWL_LEVEL */

        return s->level;

    case 0x04:	/* PWL_CTRL */

        return s->enable;

    }

    OMAP_BAD_REG(addr);

    return 0;

}
