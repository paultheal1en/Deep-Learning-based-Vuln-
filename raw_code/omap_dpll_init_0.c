static struct dpll_ctl_s  *omap_dpll_init(MemoryRegion *memory,

                           target_phys_addr_t base, omap_clk clk)

{

    struct dpll_ctl_s *s = g_malloc0(sizeof(*s));

    memory_region_init_io(&s->iomem, &omap_dpll_ops, s, "omap-dpll", 0x100);



    s->dpll = clk;

    omap_dpll_reset(s);



    memory_region_add_subregion(memory, base, &s->iomem);

    return s;

}
