static void omap_mpu_timer_write(void *opaque, target_phys_addr_t addr,

                                 uint64_t value, unsigned size)

{

    struct omap_mpu_timer_s *s = (struct omap_mpu_timer_s *) opaque;



    if (size != 4) {

        return omap_badwidth_write32(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* CNTL_TIMER */

        omap_timer_sync(s);

        s->enable = (value >> 5) & 1;

        s->ptv = (value >> 2) & 7;

        s->ar = (value >> 1) & 1;

        s->st = value & 1;

        omap_timer_update(s);

        return;



    case 0x04:	/* LOAD_TIM */

        s->reset_val = value;

        return;



    case 0x08:	/* READ_TIM */

        OMAP_RO_REG(addr);

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
