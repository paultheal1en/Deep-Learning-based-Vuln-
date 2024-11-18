void gic_set_priority(GICState *s, int cpu, int irq, uint8_t val)

{

    if (irq < GIC_INTERNAL) {

        s->priority1[irq][cpu] = val;

    } else {

        s->priority2[(irq) - GIC_INTERNAL] = val;

    }

}
