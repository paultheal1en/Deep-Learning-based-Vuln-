struct omap_gpmc_s *omap_gpmc_init(struct omap_mpu_state_s *mpu,

                                   hwaddr base,

                                   qemu_irq irq, qemu_irq drq)

{

    int cs;

    struct omap_gpmc_s *s = (struct omap_gpmc_s *)

            g_malloc0(sizeof(struct omap_gpmc_s));



    memory_region_init_io(&s->iomem, NULL, &omap_gpmc_ops, s, "omap-gpmc", 0x1000);

    memory_region_add_subregion(get_system_memory(), base, &s->iomem);



    s->irq = irq;

    s->drq = drq;

    s->accept_256 = cpu_is_omap3630(mpu);

    s->revision = cpu_class_omap3(mpu) ? 0x50 : 0x20;

    s->lastirq = 0;

    omap_gpmc_reset(s);



    /* We have to register a different IO memory handler for each

     * chip select region in case a NAND device is mapped there. We

     * make the region the worst-case size of 256MB and rely on the

     * container memory region in cs_map to chop it down to the actual

     * guest-requested size.

     */

    for (cs = 0; cs < 8; cs++) {

        memory_region_init_io(&s->cs_file[cs].nandiomem, NULL,

                              &omap_nand_ops,

                              &s->cs_file[cs],

                              "omap-nand",

                              256 * 1024 * 1024);

    }



    memory_region_init_io(&s->prefetch.iomem, NULL, &omap_prefetch_ops, s,

                          "omap-gpmc-prefetch", 256 * 1024 * 1024);

    return s;

}
