static struct omap_uwire_s *omap_uwire_init(MemoryRegion *system_memory,

                                            hwaddr base,

                                            qemu_irq txirq, qemu_irq rxirq,

                                            qemu_irq dma,

                                            omap_clk clk)

{

    struct omap_uwire_s *s = (struct omap_uwire_s *)

            g_malloc0(sizeof(struct omap_uwire_s));



    s->txirq = txirq;

    s->rxirq = rxirq;

    s->txdrq = dma;

    omap_uwire_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_uwire_ops, s, "omap-uwire", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);



    return s;

}
