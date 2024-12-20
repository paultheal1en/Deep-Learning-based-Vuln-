int bdrv_pwrite(BlockDriverState *bs, int64_t offset,

                const void *buf1, int count1)

{

    BlockDriver *drv = bs->drv;



    if (!drv)

        return -ENOMEDIUM;

    if (!drv->bdrv_pwrite)

        return bdrv_pwrite_em(bs, offset, buf1, count1);

    if (bdrv_wr_badreq_bytes(bs, offset, count1))

        return -EDOM;

    return drv->bdrv_pwrite(bs, offset, buf1, count1);

}
