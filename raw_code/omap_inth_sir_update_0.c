static void omap_inth_sir_update(struct omap_intr_handler_s *s, int is_fiq)

{

    int i, j, sir_intr, p_intr, p, f;

    uint32_t level;

    sir_intr = 0;

    p_intr = 255;



    /* Find the interrupt line with the highest dynamic priority.

     * Note: 0 denotes the hightest priority.

     * If all interrupts have the same priority, the default order is IRQ_N,

     * IRQ_N-1,...,IRQ_0. */

    for (j = 0; j < s->nbanks; ++j) {

        level = s->bank[j].irqs & ~s->bank[j].mask &

                (is_fiq ? s->bank[j].fiq : ~s->bank[j].fiq);

        for (f = ffs(level), i = f - 1, level >>= f - 1; f; i += f,

                        level >>= f) {

            p = s->bank[j].priority[i];

            if (p <= p_intr) {

                p_intr = p;

                sir_intr = 32 * j + i;

            }

            f = ffs(level >> 1);

        }

    }

    s->sir_intr[is_fiq] = sir_intr;

}
