static inline void write_IRQreg (openpic_t *opp, int n_IRQ,

                                 uint32_t reg, uint32_t val)

{

    uint32_t tmp;



    switch (reg) {

    case IRQ_IPVP:

        /* NOTE: not fully accurate for special IRQs, but simple and

           sufficient */

        /* ACTIVITY bit is read-only */

        opp->src[n_IRQ].ipvp =

            (opp->src[n_IRQ].ipvp & 0x40000000) |

            (val & 0x800F00FF);

        openpic_update_irq(opp, n_IRQ);

        DPRINTF("Set IPVP %d to 0x%08x -> 0x%08x\n",

                n_IRQ, val, opp->src[n_IRQ].ipvp);

        break;

    case IRQ_IDE:

        tmp = val & 0xC0000000;

        tmp |= val & ((1 << MAX_CPU) - 1);

        opp->src[n_IRQ].ide = tmp;

        DPRINTF("Set IDE %d to 0x%08x\n", n_IRQ, opp->src[n_IRQ].ide);

        break;

    }

}
