static void openpic_update_irq(openpic_t *opp, int n_IRQ)

{

    IRQ_src_t *src;

    int i;



    src = &opp->src[n_IRQ];



    if (!src->pending) {

        /* no irq pending */

        DPRINTF("%s: IRQ %d is not pending\n", __func__, n_IRQ);

        return;

    }

    if (test_bit(&src->ipvp, IPVP_MASK)) {

        /* Interrupt source is disabled */

        DPRINTF("%s: IRQ %d is disabled\n", __func__, n_IRQ);

        return;

    }

    if (IPVP_PRIORITY(src->ipvp) == 0) {

        /* Priority set to zero */

        DPRINTF("%s: IRQ %d has 0 priority\n", __func__, n_IRQ);

        return;

    }

    if (test_bit(&src->ipvp, IPVP_ACTIVITY)) {

        /* IRQ already active */

        DPRINTF("%s: IRQ %d is already active\n", __func__, n_IRQ);

        return;

    }

    if (src->ide == 0x00000000) {

        /* No target */

        DPRINTF("%s: IRQ %d has no target\n", __func__, n_IRQ);

        return;

    }



    if (src->ide == (1 << src->last_cpu)) {

        /* Only one CPU is allowed to receive this IRQ */

        IRQ_local_pipe(opp, src->last_cpu, n_IRQ);

    } else if (!test_bit(&src->ipvp, IPVP_MODE)) {

        /* Directed delivery mode */

        for (i = 0; i < opp->nb_cpus; i++) {

            if (test_bit(&src->ide, i))

                IRQ_local_pipe(opp, i, n_IRQ);

        }

    } else {

        /* Distributed delivery mode */

        for (i = src->last_cpu + 1; i != src->last_cpu; i++) {

            if (i == opp->nb_cpus)

                i = 0;

            if (test_bit(&src->ide, i)) {

                IRQ_local_pipe(opp, i, n_IRQ);

                src->last_cpu = i;

                break;

            }

        }

    }

}
