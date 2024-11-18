static int pl110_init(SysBusDevice *dev)

{

    pl110_state *s = FROM_SYSBUS(pl110_state, dev);



    memory_region_init_io(&s->iomem, &pl110_ops, s, "pl110", 0x1000);

    sysbus_init_mmio(dev, &s->iomem);

    sysbus_init_irq(dev, &s->irq);

    qdev_init_gpio_in(&s->busdev.qdev, pl110_mux_ctrl_set, 1);

    s->con = graphic_console_init(pl110_update_display,

                                  pl110_invalidate_display,

                                  NULL, NULL, s);

    return 0;

}
