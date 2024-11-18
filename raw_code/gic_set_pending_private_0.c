void gic_set_pending_private(GICState *s, int cpu, int irq)

{

    int cm = 1 << cpu;



    if (GIC_TEST_PENDING(irq, cm))

        return;



    DPRINTF("Set %d pending cpu %d\n", irq, cpu);

    GIC_SET_PENDING(irq, cm);

    gic_update(s);

}
