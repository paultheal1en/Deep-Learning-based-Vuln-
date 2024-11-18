static void nvic_recompute_state(NVICState *s)

{

    int i;

    int pend_prio = NVIC_NOEXC_PRIO;

    int active_prio = NVIC_NOEXC_PRIO;

    int pend_irq = 0;



    for (i = 1; i < s->num_irq; i++) {

        VecInfo *vec = &s->vectors[i];



        if (vec->enabled && vec->pending && vec->prio < pend_prio) {

            pend_prio = vec->prio;

            pend_irq = i;

        }

        if (vec->active && vec->prio < active_prio) {

            active_prio = vec->prio;

        }

    }



    if (active_prio > 0) {

        active_prio &= nvic_gprio_mask(s);

    }



    s->vectpending = pend_irq;

    s->exception_prio = active_prio;



    trace_nvic_recompute_state(s->vectpending, s->exception_prio);

}
