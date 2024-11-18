static int milkymist_memcard_init(SysBusDevice *dev)

{

    MilkymistMemcardState *s = MILKYMIST_MEMCARD(dev);

    DriveInfo *dinfo;

    BlockDriverState *bs;



    dinfo = drive_get_next(IF_SD);

    bs = dinfo ? blk_bs(blk_by_legacy_dinfo(dinfo)) : NULL;

    s->card = sd_init(bs, false);

    if (s->card == NULL) {

        return -1;

    }



    s->enabled = bs && bdrv_is_inserted(bs);



    memory_region_init_io(&s->regs_region, OBJECT(s), &memcard_mmio_ops, s,

            "milkymist-memcard", R_MAX * 4);

    sysbus_init_mmio(dev, &s->regs_region);



    return 0;

}
