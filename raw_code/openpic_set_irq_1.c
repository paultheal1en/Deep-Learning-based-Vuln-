static void openpic_set_irq(void *opaque, int n_IRQ, int level)

{

    OpenPICState *opp = opaque;

    IRQSource *src;



    src = &opp->src[n_IRQ];

    DPRINTF("openpic: set irq %d = %d ipvp=%08x\n",

            n_IRQ, level, src->ipvp);

    if (src->ipvp & IPVP_SENSE_MASK) {

        /* level-sensitive irq */

        src->pending = level;

        if (!level) {

            src->ipvp &= ~IPVP_ACTIVITY_MASK;

        }

    } else {

        /* edge-sensitive irq */

        if (level) {

            src->pending = 1;

        }

    }

    openpic_update_irq(opp, n_IRQ);

}
