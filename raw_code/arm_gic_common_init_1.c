static int arm_gic_common_init(SysBusDevice *dev)

{

    GICState *s = FROM_SYSBUS(GICState, dev);

    int num_irq = s->num_irq;



    if (s->num_cpu > NCPU) {

        hw_error("requested %u CPUs exceeds GIC maximum %d\n",

                 s->num_cpu, NCPU);

    }

    s->num_irq += GIC_BASE_IRQ;

    if (s->num_irq > GIC_MAXIRQ) {

        hw_error("requested %u interrupt lines exceeds GIC maximum %d\n",

                 num_irq, GIC_MAXIRQ);

    }

    /* ITLinesNumber is represented as (N / 32) - 1 (see

     * gic_dist_readb) so this is an implementation imposed

     * restriction, not an architectural one:

     */

    if (s->num_irq < 32 || (s->num_irq % 32)) {

        hw_error("%d interrupt lines unsupported: not divisible by 32\n",

                 num_irq);

    }



    register_savevm(NULL, "arm_gic", -1, 3, gic_save, gic_load, s);

    return 0;

}
