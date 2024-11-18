static uint32_t openpic_cpu_read_internal(void *opaque, hwaddr addr,

                                          int idx)

{

    openpic_t *opp = opaque;

    IRQ_src_t *src;

    IRQ_dst_t *dst;

    uint32_t retval;

    int n_IRQ;



    DPRINTF("%s: cpu %d addr " TARGET_FMT_plx "\n", __func__, idx, addr);

    retval = 0xFFFFFFFF;

    if (addr & 0xF)

        return retval;

    dst = &opp->dst[idx];

    addr &= 0xFF0;

    switch (addr) {

    case 0x00: /* Block Revision Register1 (BRR1) */

        retval = FSL_BRR1_IPID | FSL_BRR1_IPMJ | FSL_BRR1_IPMN;

        break;

    case 0x80: /* PCTP */

        retval = dst->pctp;

        break;

    case 0x90: /* WHOAMI */

        retval = idx;

        break;

    case 0xA0: /* PIAC */

        DPRINTF("Lower OpenPIC INT output\n");

        qemu_irq_lower(dst->irqs[OPENPIC_OUTPUT_INT]);

        n_IRQ = IRQ_get_next(opp, &dst->raised);

        DPRINTF("PIAC: irq=%d\n", n_IRQ);

        if (n_IRQ == -1) {

            /* No more interrupt pending */

            retval = IPVP_VECTOR(opp->spve);

        } else {

            src = &opp->src[n_IRQ];

            if (!test_bit(&src->ipvp, IPVP_ACTIVITY) ||

                !(IPVP_PRIORITY(src->ipvp) > dst->pctp)) {

                /* - Spurious level-sensitive IRQ

                 * - Priorities has been changed

                 *   and the pending IRQ isn't allowed anymore

                 */

                reset_bit(&src->ipvp, IPVP_ACTIVITY);

                retval = IPVP_VECTOR(opp->spve);

            } else {

                /* IRQ enter servicing state */

                IRQ_setbit(&dst->servicing, n_IRQ);

                retval = IPVP_VECTOR(src->ipvp);

            }

            IRQ_resetbit(&dst->raised, n_IRQ);

            dst->raised.next = -1;

            if (!test_bit(&src->ipvp, IPVP_SENSE)) {

                /* edge-sensitive IRQ */

                reset_bit(&src->ipvp, IPVP_ACTIVITY);

                src->pending = 0;

            }



            if ((n_IRQ >= opp->irq_ipi0) &&  (n_IRQ < (opp->irq_ipi0 + MAX_IPI))) {

                src->ide &= ~(1 << idx);

                if (src->ide && !test_bit(&src->ipvp, IPVP_SENSE)) {

                    /* trigger on CPUs that didn't know about it yet */

                    openpic_set_irq(opp, n_IRQ, 1);

                    openpic_set_irq(opp, n_IRQ, 0);

                    /* if all CPUs knew about it, set active bit again */

                    set_bit(&src->ipvp, IPVP_ACTIVITY);

                }

            }

        }

        break;

    case 0xB0: /* PEOI */

        retval = 0;

        break;

    default:

        break;

    }

    DPRINTF("%s: => %08x\n", __func__, retval);



    return retval;

}
