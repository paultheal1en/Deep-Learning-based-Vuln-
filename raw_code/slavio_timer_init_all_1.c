static void slavio_timer_init_all(target_phys_addr_t addr, qemu_irq master_irq,

                                  qemu_irq *cpu_irqs, unsigned int num_cpus)

{

    DeviceState *dev;

    SysBusDevice *s;

    unsigned int i;



    dev = qdev_create(NULL, "slavio_timer");

    qdev_prop_set_uint32(dev, "num_cpus", num_cpus);

    qdev_init(dev);

    s = sysbus_from_qdev(dev);

    sysbus_connect_irq(s, 0, master_irq);

    sysbus_mmio_map(s, 0, addr + SYS_TIMER_OFFSET);



    for (i = 0; i < MAX_CPUS; i++) {

        sysbus_mmio_map(s, i + 1, addr + (target_phys_addr_t)CPU_TIMER_OFFSET(i));

        sysbus_connect_irq(s, i + 1, cpu_irqs[i]);

    }

}
