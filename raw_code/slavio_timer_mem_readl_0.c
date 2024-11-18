static uint32_t slavio_timer_mem_readl(void *opaque, target_phys_addr_t addr)

{

    SLAVIO_TIMERState *s = opaque;

    uint32_t saddr, ret;



    saddr = (addr & TIMER_MAXADDR) >> 2;

    switch (saddr) {

    case 0:

        // read limit (system counter mode) or read most signifying

        // part of counter (user mode)

        if (slavio_timer_is_user(s)) {

            // read user timer MSW

            slavio_timer_get_out(s);

            ret = s->counthigh;

        } else {

            // read limit

            // clear irq

            qemu_irq_lower(s->irq);

            s->reached = 0;

            ret = s->limit & 0x7fffffff;

        }

        break;

    case 1:

        // read counter and reached bit (system mode) or read lsbits

        // of counter (user mode)

        slavio_timer_get_out(s);

        if (slavio_timer_is_user(s)) // read user timer LSW

            ret = s->count & 0xffffffe00;

        else // read limit

            ret = (s->count & 0x7ffffe00) | s->reached;

        break;

    case 3:

        // only available in processor counter/timer

        // read start/stop status

        ret = s->running;

        break;

    case 4:

        // only available in system counter

        // read user/system mode

        ret = s->slave_mode;

        break;

    default:

        DPRINTF("invalid read address " TARGET_FMT_plx "\n", addr);

        ret = 0;

        break;

    }

    DPRINTF("read " TARGET_FMT_plx " = %08x\n", addr, ret);



    return ret;

}
