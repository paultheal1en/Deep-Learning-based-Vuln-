static int omap2_intc_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    struct omap_intr_handler_s *s = OMAP_INTC(dev);



    if (!s->iclk) {

        hw_error("omap2-intc: iclk not connected\n");

    }

    if (!s->fclk) {

        hw_error("omap2-intc: fclk not connected\n");

    }

    s->level_only = 1;

    s->nbanks = 3;

    sysbus_init_irq(sbd, &s->parent_intr[0]);

    sysbus_init_irq(sbd, &s->parent_intr[1]);

    qdev_init_gpio_in(dev, omap_set_intr_noedge, s->nbanks * 32);

    memory_region_init_io(&s->mmio, OBJECT(s), &omap2_inth_mem_ops, s,

                          "omap2-intc", 0x1000);

    sysbus_init_mmio(sbd, &s->mmio);

    return 0;

}
