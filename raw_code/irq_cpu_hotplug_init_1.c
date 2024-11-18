static int irq_cpu_hotplug_init(SCLPEvent *event)

{

    irq_cpu_hotplug = *qemu_allocate_irqs(trigger_signal, event, 1);

    return 0;

}
