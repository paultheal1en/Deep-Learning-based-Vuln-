static int omap2_gpio_init(SysBusDevice *sbd)

{

    DeviceState *dev = DEVICE(sbd);

    struct omap2_gpif_s *s = OMAP2_GPIO(dev);

    int i;



    if (!s->iclk) {

        hw_error("omap2-gpio: iclk not connected\n");

    }

    if (s->mpu_model < omap3430) {

        s->modulecount = (s->mpu_model < omap2430) ? 4 : 5;

        memory_region_init_io(&s->iomem, OBJECT(s), &omap2_gpif_top_ops, s,

                              "omap2.gpio", 0x1000);

        sysbus_init_mmio(sbd, &s->iomem);

    } else {

        s->modulecount = 6;

    }

    s->modules = g_new0(struct omap2_gpio_s, s->modulecount);

    s->handler = g_new0(qemu_irq, s->modulecount * 32);

    qdev_init_gpio_in(dev, omap2_gpio_set, s->modulecount * 32);

    qdev_init_gpio_out(dev, s->handler, s->modulecount * 32);

    for (i = 0; i < s->modulecount; i++) {

        struct omap2_gpio_s *m = &s->modules[i];

        if (!s->fclk[i]) {

            hw_error("omap2-gpio: fclk%d not connected\n", i);

        }

        m->revision = (s->mpu_model < omap3430) ? 0x18 : 0x25;

        m->handler = &s->handler[i * 32];

        sysbus_init_irq(sbd, &m->irq[0]); /* mpu irq */

        sysbus_init_irq(sbd, &m->irq[1]); /* dsp irq */

        sysbus_init_irq(sbd, &m->wkup);

        memory_region_init_io(&m->iomem, OBJECT(s), &omap2_gpio_module_ops, m,

                              "omap.gpio-module", 0x1000);

        sysbus_init_mmio(sbd, &m->iomem);

    }

    return 0;

}
