struct omap_mcbsp_s *omap_mcbsp_init(MemoryRegion *system_memory,

                target_phys_addr_t base,

                qemu_irq *irq, qemu_irq *dma, omap_clk clk)

{

    struct omap_mcbsp_s *s = (struct omap_mcbsp_s *)

            g_malloc0(sizeof(struct omap_mcbsp_s));



    s->txirq = irq[0];

    s->rxirq = irq[1];

    s->txdrq = dma[0];

    s->rxdrq = dma[1];

    s->sink_timer = qemu_new_timer_ns(vm_clock, omap_mcbsp_sink_tick, s);

    s->source_timer = qemu_new_timer_ns(vm_clock, omap_mcbsp_source_tick, s);

    omap_mcbsp_reset(s);



    memory_region_init_io(&s->iomem, &omap_mcbsp_ops, s, "omap-mcbsp", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
