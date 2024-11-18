struct omap_sdrc_s *omap_sdrc_init(MemoryRegion *sysmem,

                                   hwaddr base)

{

    struct omap_sdrc_s *s = (struct omap_sdrc_s *)

            g_malloc0(sizeof(struct omap_sdrc_s));



    omap_sdrc_reset(s);



    memory_region_init_io(&s->iomem, NULL, &omap_sdrc_ops, s, "omap.sdrc", 0x1000);

    memory_region_add_subregion(sysmem, base, &s->iomem);



    return s;

}
