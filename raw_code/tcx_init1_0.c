static int tcx_init1(SysBusDevice *dev)

{

    TCXState *s = FROM_SYSBUS(TCXState, dev);

    ram_addr_t vram_offset = 0;

    int size;

    uint8_t *vram_base;



    memory_region_init_ram(&s->vram_mem, "tcx.vram",

                           s->vram_size * (1 + 4 + 4));

    vmstate_register_ram_global(&s->vram_mem);

    vram_base = memory_region_get_ram_ptr(&s->vram_mem);



    /* 8-bit plane */

    s->vram = vram_base;

    size = s->vram_size;

    memory_region_init_alias(&s->vram_8bit, "tcx.vram.8bit",

                             &s->vram_mem, vram_offset, size);

    sysbus_init_mmio(dev, &s->vram_8bit);

    vram_offset += size;

    vram_base += size;



    /* DAC */

    memory_region_init_io(&s->dac, &tcx_dac_ops, s, "tcx.dac", TCX_DAC_NREGS);

    sysbus_init_mmio(dev, &s->dac);



    /* TEC (dummy) */

    memory_region_init_io(&s->tec, &dummy_ops, s, "tcx.tec", TCX_TEC_NREGS);

    sysbus_init_mmio(dev, &s->tec);

    /* THC: NetBSD writes here even with 8-bit display: dummy */

    memory_region_init_io(&s->thc24, &dummy_ops, s, "tcx.thc24",

                          TCX_THC_NREGS_24);

    sysbus_init_mmio(dev, &s->thc24);



    if (s->depth == 24) {

        /* 24-bit plane */

        size = s->vram_size * 4;

        s->vram24 = (uint32_t *)vram_base;

        s->vram24_offset = vram_offset;

        memory_region_init_alias(&s->vram_24bit, "tcx.vram.24bit",

                                 &s->vram_mem, vram_offset, size);

        sysbus_init_mmio(dev, &s->vram_24bit);

        vram_offset += size;

        vram_base += size;



        /* Control plane */

        size = s->vram_size * 4;

        s->cplane = (uint32_t *)vram_base;

        s->cplane_offset = vram_offset;

        memory_region_init_alias(&s->vram_cplane, "tcx.vram.cplane",

                                 &s->vram_mem, vram_offset, size);

        sysbus_init_mmio(dev, &s->vram_cplane);



        s->con = graphic_console_init(tcx24_update_display,

                                      tcx24_invalidate_display,

                                      tcx24_screen_dump, NULL, s);

    } else {

        /* THC 8 bit (dummy) */

        memory_region_init_io(&s->thc8, &dummy_ops, s, "tcx.thc8",

                              TCX_THC_NREGS_8);

        sysbus_init_mmio(dev, &s->thc8);



        s->con = graphic_console_init(tcx_update_display,

                                      tcx_invalidate_display,

                                      tcx_screen_dump, NULL, s);

    }



    qemu_console_resize(s->con, s->width, s->height);

    return 0;

}
