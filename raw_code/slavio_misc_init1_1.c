static int slavio_misc_init1(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    MiscState *s = SLAVIO_MISC(dev);



    sysbus_init_irq(sbd, &s->irq);

    sysbus_init_irq(sbd, &s->fdc_tc);



    /* 8 bit registers */

    /* Slavio control */

    memory_region_init_io(&s->cfg_iomem, OBJECT(s), &slavio_cfg_mem_ops, s,

                          "configuration", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->cfg_iomem);



    /* Diagnostics */

    memory_region_init_io(&s->diag_iomem, OBJECT(s), &slavio_diag_mem_ops, s,

                          "diagnostic", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->diag_iomem);



    /* Modem control */

    memory_region_init_io(&s->mdm_iomem, OBJECT(s), &slavio_mdm_mem_ops, s,

                          "modem", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->mdm_iomem);



    /* 16 bit registers */

    /* ss600mp diag LEDs */

    memory_region_init_io(&s->led_iomem, OBJECT(s), &slavio_led_mem_ops, s,

                          "leds", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->led_iomem);



    /* 32 bit registers */

    /* System control */

    memory_region_init_io(&s->sysctrl_iomem, OBJECT(s), &slavio_sysctrl_mem_ops, s,

                          "system-control", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->sysctrl_iomem);



    /* AUX 1 (Misc System Functions) */

    memory_region_init_io(&s->aux1_iomem, OBJECT(s), &slavio_aux1_mem_ops, s,

                          "misc-system-functions", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->aux1_iomem);



    /* AUX 2 (Software Powerdown Control) */

    memory_region_init_io(&s->aux2_iomem, OBJECT(s), &slavio_aux2_mem_ops, s,

                          "software-powerdown-control", MISC_SIZE);

    sysbus_init_mmio(sbd, &s->aux2_iomem);



    qdev_init_gpio_in(dev, slavio_set_power_fail, 1);



    return 0;

}
