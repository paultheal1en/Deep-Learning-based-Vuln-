static int milkymist_vgafb_init(SysBusDevice *dev)

{

    MilkymistVgafbState *s = FROM_SYSBUS(typeof(*s), dev);



    memory_region_init_io(&s->regs_region, &vgafb_mmio_ops, s,

            "milkymist-vgafb", R_MAX * 4);

    sysbus_init_mmio(dev, &s->regs_region);



    s->con = graphic_console_init(vgafb_update_display,

                                  vgafb_invalidate_display,

                                  NULL, NULL, s);



    return 0;

}
