static struct omap_rtc_s *omap_rtc_init(target_phys_addr_t base,

                qemu_irq *irq, omap_clk clk)

{

    int iomemtype;

    struct omap_rtc_s *s = (struct omap_rtc_s *)

            qemu_mallocz(sizeof(struct omap_rtc_s));



    s->irq = irq[0];

    s->alarm = irq[1];

    s->clk = qemu_new_timer(rt_clock, omap_rtc_tick, s);



    omap_rtc_reset(s);



    iomemtype = cpu_register_io_memory(omap_rtc_readfn,

                    omap_rtc_writefn, s, DEVICE_NATIVE_ENDIAN);

    cpu_register_physical_memory(base, 0x800, iomemtype);



    return s;

}
