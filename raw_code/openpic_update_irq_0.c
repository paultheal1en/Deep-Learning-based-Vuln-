static void openpic_update_irq(OpenPICState *opp, int n_IRQ)

{

    IRQSource *src;

    bool active, was_active;

    int i;



    src = &opp->src[n_IRQ];

    active = src->pending;



    if ((src->ivpr & IVPR_MASK_MASK) && !src->nomask) {

        /* Interrupt source is disabled */

        DPRINTF("%s: IRQ %d is disabled\n", __func__, n_IRQ);

        active = false;

    }



    was_active = !!(src->ivpr & IVPR_ACTIVITY_MASK);



    /*

     * We don't have a similar check for already-active because

     * ctpr may have changed and we need to withdraw the interrupt.

     */

    if (!active && !was_active) {

        DPRINTF("%s: IRQ %d is already inactive\n", __func__, n_IRQ);

        return;

    }



    if (active) {

        src->ivpr |= IVPR_ACTIVITY_MASK;

    } else {

        src->ivpr &= ~IVPR_ACTIVITY_MASK;

    }



    if (src->idr == 0) {

        /* No target */

        DPRINTF("%s: IRQ %d has no target\n", __func__, n_IRQ);

        return;

    }



    if (src->idr == (1 << src->last_cpu)) {

        /* Only one CPU is allowed to receive this IRQ */

        IRQ_local_pipe(opp, src->last_cpu, n_IRQ, active, was_active);

    } else if (!(src->ivpr & IVPR_MODE_MASK)) {

        /* Directed delivery mode */

        for (i = 0; i < opp->nb_cpus; i++) {

            if (src->destmask & (1 << i)) {

                IRQ_local_pipe(opp, i, n_IRQ, active, was_active);

            }

        }

    } else {

        /* Distributed delivery mode */

        for (i = src->last_cpu + 1; i != src->last_cpu; i++) {

            if (i == opp->nb_cpus) {

                i = 0;

            }

            if (src->destmask & (1 << i)) {

                IRQ_local_pipe(opp, i, n_IRQ, active, was_active);

                src->last_cpu = i;

                break;

            }

        }

    }

}
