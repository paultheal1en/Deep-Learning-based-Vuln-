static int macio_newworld_initfn(PCIDevice *d)

{

    MacIOState *s = MACIO(d);

    NewWorldMacIOState *ns = NEWWORLD_MACIO(d);

    SysBusDevice *sysbus_dev;

    MemoryRegion *timer_memory = g_new(MemoryRegion, 1);

    int i;

    int cur_irq = 0;

    int ret = macio_common_initfn(d);

    if (ret < 0) {

        return ret;

    }



    sysbus_dev = SYS_BUS_DEVICE(&s->cuda);

    sysbus_connect_irq(sysbus_dev, 0, ns->irqs[cur_irq++]);



    if (s->pic_mem) {

        /* OpenPIC */

        memory_region_add_subregion(&s->bar, 0x40000, s->pic_mem);

    }



    /* IDE buses */

    for (i = 0; i < ARRAY_SIZE(ns->ide); i++) {

        qemu_irq irq0 = ns->irqs[cur_irq++];

        qemu_irq irq1 = ns->irqs[cur_irq++];



        ret = macio_initfn_ide(s, &ns->ide[i], irq0, irq1, 0x16 + (i * 4));

        if (ret < 0) {

            return ret;

        }

    }



    /* Timer */

    memory_region_init_io(timer_memory, OBJECT(s), &timer_ops, NULL, "timer",

                          0x1000);

    memory_region_add_subregion(&s->bar, 0x15000, timer_memory);



    return 0;

}
