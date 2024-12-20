int bdrv_pread(BlockDriverState *bs, int64_t offset,
               void *buf1, int count1)
{
    BlockDriver *drv = bs->drv;
    if (!drv)
        return -ENOMEDIUM;
    if (!drv->bdrv_pread)
        return bdrv_pread_em(bs, offset, buf1, count1);
    return drv->bdrv_pread(bs, offset, buf1, count1);
}