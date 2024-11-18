qemu_irq qemu_irq_invert(qemu_irq irq)

{

    /* The default state for IRQs is low, so raise the output now.  */

    qemu_irq_raise(irq);

    return qemu_allocate_irqs(qemu_notirq, irq, 1)[0];

}
