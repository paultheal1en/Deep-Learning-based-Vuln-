static int sl_nand_init(SysBusDevice *dev)

{

    SLNANDState *s = SL_NAND(dev);

    DriveInfo *nand;



    s->ctl = 0;

    nand = drive_get(IF_MTD, 0, 0);

    s->nand = nand_init(nand ? blk_bs(blk_by_legacy_dinfo(nand)) : NULL,

                        s->manf_id, s->chip_id);



    memory_region_init_io(&s->iomem, OBJECT(s), &sl_ops, s, "sl", 0x40);

    sysbus_init_mmio(dev, &s->iomem);



    return 0;

}
