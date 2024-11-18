StrongARMState *sa1110_init(MemoryRegion *sysmem,

                            unsigned int sdram_size, const char *rev)

{

    StrongARMState *s;

    int i;



    s = g_new0(StrongARMState, 1);



    if (!rev) {

        rev = "sa1110-b5";

    }



    if (strncmp(rev, "sa1110", 6)) {

        error_report("Machine requires a SA1110 processor.");

        exit(1);

    }



    s->cpu = ARM_CPU(cpu_generic_init(TYPE_ARM_CPU, rev));



    if (!s->cpu) {

        error_report("Unable to find CPU definition");

        exit(1);

    }



    memory_region_allocate_system_memory(&s->sdram, NULL, "strongarm.sdram",

                                         sdram_size);

    memory_region_add_subregion(sysmem, SA_SDCS0, &s->sdram);



    s->pic = sysbus_create_varargs("strongarm_pic", 0x90050000,

                    qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_IRQ),

                    qdev_get_gpio_in(DEVICE(s->cpu), ARM_CPU_FIQ),

                    NULL);



    sysbus_create_varargs("pxa25x-timer", 0x90000000,

                    qdev_get_gpio_in(s->pic, SA_PIC_OSTC0),

                    qdev_get_gpio_in(s->pic, SA_PIC_OSTC1),

                    qdev_get_gpio_in(s->pic, SA_PIC_OSTC2),

                    qdev_get_gpio_in(s->pic, SA_PIC_OSTC3),

                    NULL);



    sysbus_create_simple(TYPE_STRONGARM_RTC, 0x90010000,

                    qdev_get_gpio_in(s->pic, SA_PIC_RTC_ALARM));



    s->gpio = strongarm_gpio_init(0x90040000, s->pic);



    s->ppc = sysbus_create_varargs(TYPE_STRONGARM_PPC, 0x90060000, NULL);



    for (i = 0; sa_serial[i].io_base; i++) {

        DeviceState *dev = qdev_create(NULL, TYPE_STRONGARM_UART);

        qdev_prop_set_chr(dev, "chardev", serial_hds[i]);

        qdev_init_nofail(dev);

        sysbus_mmio_map(SYS_BUS_DEVICE(dev), 0,

                sa_serial[i].io_base);

        sysbus_connect_irq(SYS_BUS_DEVICE(dev), 0,

                qdev_get_gpio_in(s->pic, sa_serial[i].irq));

    }



    s->ssp = sysbus_create_varargs(TYPE_STRONGARM_SSP, 0x80070000,

                qdev_get_gpio_in(s->pic, SA_PIC_SSP), NULL);

    s->ssp_bus = (SSIBus *)qdev_get_child_bus(s->ssp, "ssi");



    return s;

}
