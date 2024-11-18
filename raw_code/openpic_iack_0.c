static uint32_t openpic_iack(OpenPICState *opp, IRQDest *dst, int cpu)

{

    IRQSource *src;

    int retval, irq;



    DPRINTF("Lower OpenPIC INT output\n");

    qemu_irq_lower(dst->irqs[OPENPIC_OUTPUT_INT]);



    irq = IRQ_get_next(opp, &dst->raised);

    DPRINTF("IACK: irq=%d\n", irq);



    if (irq == -1) {

        /* No more interrupt pending */

        return opp->spve;

    }



    src = &opp->src[irq];

    if (!(src->ivpr & IVPR_ACTIVITY_MASK) ||

            !(IVPR_PRIORITY(src->ivpr) > dst->ctpr)) {

        /* - Spurious level-sensitive IRQ

         * - Priorities has been changed

         *   and the pending IRQ isn't allowed anymore

         */

        src->ivpr &= ~IVPR_ACTIVITY_MASK;

        retval = opp->spve;

    } else {

        /* IRQ enter servicing state */

        IRQ_setbit(&dst->servicing, irq);

        retval = IVPR_VECTOR(opp, src->ivpr);

    }

    IRQ_resetbit(&dst->raised, irq);

    if (!src->level) {

        /* edge-sensitive IRQ */

        src->ivpr &= ~IVPR_ACTIVITY_MASK;

        src->pending = 0;

    }



    if ((irq >= opp->irq_ipi0) &&  (irq < (opp->irq_ipi0 + MAX_IPI))) {

        src->idr &= ~(1 << cpu);

        if (src->idr && !src->level) {

            /* trigger on CPUs that didn't know about it yet */

            openpic_set_irq(opp, irq, 1);

            openpic_set_irq(opp, irq, 0);

            /* if all CPUs knew about it, set active bit again */

            src->ivpr |= IVPR_ACTIVITY_MASK;

        }

    }



    return retval;

}
