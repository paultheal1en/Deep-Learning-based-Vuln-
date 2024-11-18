int bdrv_get_info(BlockDriverState *bs, BlockDriverInfo *bdi)

{

    BlockDriver *drv = bs->drv;

    if (!drv)

        return -ENOMEDIUM;

    if (!drv->bdrv_get_info)

        return -ENOTSUP;

    memset(bdi, 0, sizeof(*bdi));

    return drv->bdrv_get_info(bs, bdi);

}
