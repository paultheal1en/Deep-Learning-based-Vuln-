static uint64_t m5208_timer_read(void *opaque, target_phys_addr_t addr,

                                 unsigned size)

{

    m5208_timer_state *s = (m5208_timer_state *)opaque;

    switch (addr) {

    case 0:

        return s->pcsr;

    case 2:

        return s->pmr;

    case 4:

        return ptimer_get_count(s->timer);

    default:

        hw_error("m5208_timer_read: Bad offset 0x%x\n", (int)addr);

        return 0;

    }

}
