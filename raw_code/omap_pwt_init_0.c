static struct omap_pwt_s *omap_pwt_init(MemoryRegion *system_memory,

                                        target_phys_addr_t base,

                                        omap_clk clk)

{

    struct omap_pwt_s *s = g_malloc0(sizeof(*s));

    s->clk = clk;

    omap_pwt_reset(s);



    memory_region_init_io(&s->iomem, &omap_pwt_ops, s,

                          "omap-pwt", 0x800);

    memory_region_add_subregion(system_memory, base, &s->iomem);

    return s;

}
