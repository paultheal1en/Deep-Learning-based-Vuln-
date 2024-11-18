static void rc4030_realize(DeviceState *dev, Error **errp)

{

    rc4030State *s = RC4030(dev);

    Object *o = OBJECT(dev);

    int i;



    s->periodic_timer = timer_new_ns(QEMU_CLOCK_VIRTUAL,

                                     rc4030_periodic_timer, s);



    memory_region_init_io(&s->iomem_chipset, NULL, &rc4030_ops, s,

                          "rc4030.chipset", 0x300);

    memory_region_init_io(&s->iomem_jazzio, NULL, &jazzio_ops, s,

                          "rc4030.jazzio", 0x00001000);



    memory_region_init_rom_device(&s->dma_tt, o,

                                  &rc4030_dma_tt_ops, s, "dma-table",

                                  MAX_TL_ENTRIES * sizeof(dma_pagetable_entry),

                                  NULL);

    memory_region_init(&s->dma_tt_alias, o, "dma-table-alias", 0);

    memory_region_init(&s->dma_mr, o, "dma", INT32_MAX);

    for (i = 0; i < MAX_TL_ENTRIES; ++i) {

        memory_region_init_alias(&s->dma_mrs[i], o, "dma-alias",

                                 get_system_memory(), 0, DMA_PAGESIZE);

        memory_region_set_enabled(&s->dma_mrs[i], false);

        memory_region_add_subregion(&s->dma_mr, i * DMA_PAGESIZE,

                                    &s->dma_mrs[i]);

    }

    address_space_init(&s->dma_as, &s->dma_mr, "rc4030-dma");

}
