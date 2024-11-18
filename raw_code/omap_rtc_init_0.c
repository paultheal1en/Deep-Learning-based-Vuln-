static struct omap_rtc_s *omap_rtc_init(MemoryRegion *system_memory,

                target_phys_addr_t base,

                qemu_irq *irq, omap_clk clk)

{

    struct omap_rtc_s *s = (struct omap_rtc_s *)

            g_malloc0(sizeof(struct omap_rtc_s));



    s->irq = irq[0];

    s->alarm = irq[1];

    s->clk = qemu_new_timer_ms(rt_clock, omap_rtc_tick, s);



    omap_rtc_reset(s);



    memory_region_init_io(&s->iomem, &omap_rtc_ops, s,

                          "omap-rtc", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
