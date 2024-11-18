static void armv7m_nvic_clear_pending(void *opaque, int irq)

{

    NVICState *s = (NVICState *)opaque;

    VecInfo *vec;



    assert(irq > ARMV7M_EXCP_RESET && irq < s->num_irq);



    vec = &s->vectors[irq];

    trace_nvic_clear_pending(irq, vec->enabled, vec->prio);

    if (vec->pending) {

        vec->pending = 0;

        nvic_irq_update(s);

    }

}
