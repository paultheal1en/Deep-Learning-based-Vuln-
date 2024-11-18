int bdrv_is_inserted(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    int ret;

    if (!drv)

        return 0;

    if (!drv->bdrv_is_inserted)

        return !bs->tray_open;

    ret = drv->bdrv_is_inserted(bs);

    return ret;

}
