static void m5208_timer_write(void *opaque, target_phys_addr_t offset,

                              uint64_t value, unsigned size)

{

    m5208_timer_state *s = (m5208_timer_state *)opaque;

    int prescale;

    int limit;

    switch (offset) {

    case 0:

        /* The PIF bit is set-to-clear.  */

        if (value & PCSR_PIF) {

            s->pcsr &= ~PCSR_PIF;

            value &= ~PCSR_PIF;

        }

        /* Avoid frobbing the timer if we're just twiddling IRQ bits. */

        if (((s->pcsr ^ value) & ~PCSR_PIE) == 0) {

            s->pcsr = value;

            m5208_timer_update(s);

            return;

        }



        if (s->pcsr & PCSR_EN)

            ptimer_stop(s->timer);



        s->pcsr = value;



        prescale = 1 << ((s->pcsr & PCSR_PRE_MASK) >> PCSR_PRE_SHIFT);

        ptimer_set_freq(s->timer, (SYS_FREQ / 2) / prescale);

        if (s->pcsr & PCSR_RLD)

            limit = s->pmr;

        else

            limit = 0xffff;

        ptimer_set_limit(s->timer, limit, 0);



        if (s->pcsr & PCSR_EN)

            ptimer_run(s->timer, 0);

        break;

    case 2:

        s->pmr = value;

        s->pcsr &= ~PCSR_PIF;

        if ((s->pcsr & PCSR_RLD) == 0) {

            if (s->pcsr & PCSR_OVW)

                ptimer_set_count(s->timer, value);

        } else {

            ptimer_set_limit(s->timer, value, s->pcsr & PCSR_OVW);

        }

        break;

    case 4:

        break;

    default:

        hw_error("m5208_timer_write: Bad offset 0x%x\n", (int)offset);

        break;

    }

    m5208_timer_update(s);

}
