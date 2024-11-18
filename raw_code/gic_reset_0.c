static void gic_reset(gic_state *s)

{

    int i;

    memset(s->irq_state, 0, GIC_NIRQ * sizeof(gic_irq_state));

    for (i = 0 ; i < NUM_CPU(s); i++) {

        s->priority_mask[i] = 0xf0;

        s->current_pending[i] = 1023;

        s->running_irq[i] = 1023;

        s->running_priority[i] = 0x100;

#ifdef NVIC

        /* The NVIC doesn't have per-cpu interfaces, so enable by default.  */

        s->cpu_enabled[i] = 1;

#else

        s->cpu_enabled[i] = 0;

#endif

    }

    for (i = 0; i < 16; i++) {

        GIC_SET_ENABLED(i);

        GIC_SET_TRIGGER(i);

    }

#ifdef NVIC

    /* The NVIC is always enabled.  */

    s->enabled = 1;

#else

    s->enabled = 0;

#endif

}
