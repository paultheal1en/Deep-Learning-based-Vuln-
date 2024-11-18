int bdrv_load_vmstate(BlockDriverState *bs, uint8_t *buf,

                      int64_t pos, int size)

{

    BlockDriver *drv = bs->drv;

    if (!drv)

        return -ENOMEDIUM;

    if (drv->bdrv_load_vmstate)

        return drv->bdrv_load_vmstate(bs, buf, pos, size);

    if (bs->file)

        return bdrv_load_vmstate(bs->file, buf, pos, size);

    return -ENOTSUP;

}
