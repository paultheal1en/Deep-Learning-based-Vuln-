void armv7m_nvic_complete_irq(void *opaque, int irq)

{

    nvic_state *s = (nvic_state *)opaque;

    if (irq >= 16)

        irq += 16;

    gic_complete_irq(&s->gic, 0, irq);

}
