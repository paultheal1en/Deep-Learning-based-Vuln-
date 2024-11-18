static void omap_wd_timer_write(void *opaque, target_phys_addr_t addr,

                                uint64_t value, unsigned size)

{

    struct omap_watchdog_timer_s *s = (struct omap_watchdog_timer_s *) opaque;



    if (size != 2) {

        return omap_badwidth_write16(opaque, addr, value);

    }



    switch (addr) {

    case 0x00:	/* CNTL_TIMER */

        omap_timer_sync(&s->timer);

        s->timer.ptv = (value >> 9) & 7;

        s->timer.ar = (value >> 8) & 1;

        s->timer.st = (value >> 7) & 1;

        s->free = (value >> 1) & 1;

        omap_timer_update(&s->timer);

        break;



    case 0x04:	/* LOAD_TIMER */

        s->timer.reset_val = value & 0xffff;

        break;



    case 0x08:	/* TIMER_MODE */

        if (!s->mode && ((value >> 15) & 1))

            omap_clk_get(s->timer.clk);

        s->mode |= (value >> 15) & 1;

        if (s->last_wr == 0xf5) {

            if ((value & 0xff) == 0xa0) {

                if (s->mode) {

                    s->mode = 0;

                    omap_clk_put(s->timer.clk);

                }

            } else {

                /* XXX: on T|E hardware somehow this has no effect,

                 * on Zire 71 it works as specified.  */

                s->reset = 1;

                qemu_system_reset_request();

            }

        }

        s->last_wr = value & 0xff;

        break;



    default:

        OMAP_BAD_REG(addr);

    }

}
