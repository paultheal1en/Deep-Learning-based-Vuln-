static struct omap_mpuio_s *omap_mpuio_init(MemoryRegion *memory,

                hwaddr base,

                qemu_irq kbd_int, qemu_irq gpio_int, qemu_irq wakeup,

                omap_clk clk)

{

    struct omap_mpuio_s *s = (struct omap_mpuio_s *)

            g_malloc0(sizeof(struct omap_mpuio_s));



    s->irq = gpio_int;

    s->kbd_irq = kbd_int;

    s->wakeup = wakeup;

    s->in = qemu_allocate_irqs(omap_mpuio_set, s, 16);

    omap_mpuio_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_mpuio_ops, s,

                          "omap-mpuio", 0x800);

    memory_region_add_subregion(memory, base, &s->iomem);



    omap_clk_adduser(clk, qemu_allocate_irq(omap_mpuio_onoff, s, 0));



    return s;

}
