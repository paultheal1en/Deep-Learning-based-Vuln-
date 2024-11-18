static void slavio_timer_irq(void *opaque)

{

    TimerContext *tc = opaque;

    SLAVIO_TIMERState *s = tc->s;

    CPUTimerState *t = &s->cputimer[tc->timer_index];



    slavio_timer_get_out(t);

    DPRINTF("callback: count %x%08x\n", t->counthigh, t->count);

    t->reached = TIMER_REACHED;

    /* there is no interrupt if user timer or free-run */

    if (!slavio_timer_is_user(tc) && t->limit != 0) {

        qemu_irq_raise(t->irq);

    }

}
