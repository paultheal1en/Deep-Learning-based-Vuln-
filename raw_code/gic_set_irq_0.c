static void gic_set_irq(void *opaque, int irq, int level)

{

    gic_state *s = (gic_state *)opaque;

    /* The first external input line is internal interrupt 32.  */

    irq += GIC_INTERNAL;

    if (level == GIC_TEST_LEVEL(irq, ALL_CPU_MASK))

        return;



    if (level) {

        GIC_SET_LEVEL(irq, ALL_CPU_MASK);

        if (GIC_TEST_TRIGGER(irq) || GIC_TEST_ENABLED(irq, ALL_CPU_MASK)) {

            DPRINTF("Set %d pending mask %x\n", irq, GIC_TARGET(irq));

            GIC_SET_PENDING(irq, GIC_TARGET(irq));

        }

    } else {

        GIC_CLEAR_LEVEL(irq, ALL_CPU_MASK);

    }

    gic_update(s);

}
