static void translate_priority(GICState *s, int irq, int cpu,

                               uint32_t *field, bool to_kernel)

{

    if (to_kernel) {

        *field = GIC_GET_PRIORITY(irq, cpu) & 0xff;

    } else {

        gic_set_priority(s, cpu, irq, *field & 0xff);

    }

}
