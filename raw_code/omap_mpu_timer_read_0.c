static uint64_t omap_mpu_timer_read(void *opaque, target_phys_addr_t addr,

                                    unsigned size)

{

    struct omap_mpu_timer_s *s = (struct omap_mpu_timer_s *) opaque;



    if (size != 4) {

        return omap_badwidth_read32(opaque, addr);

    }



    switch (addr) {

    case 0x00:	/* CNTL_TIMER */

        return (s->enable << 5) | (s->ptv << 2) | (s->ar << 1) | s->st;



    case 0x04:	/* LOAD_TIM */

        break;



    case 0x08:	/* READ_TIM */

        return omap_timer_read(s);

    }



    OMAP_BAD_REG(addr);

    return 0;

}
