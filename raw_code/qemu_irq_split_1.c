qemu_irq qemu_irq_split(qemu_irq irq1, qemu_irq irq2)

{

    qemu_irq *s = g_malloc0(2 * sizeof(qemu_irq));

    s[0] = irq1;

    s[1] = irq2;

    return qemu_allocate_irqs(qemu_splitirq, s, 1)[0];

}
