bdrv_driver_pwritev_compressed(BlockDriverState *bs, uint64_t offset,
                               uint64_t bytes, QEMUIOVector *qiov)
{
    BlockDriver *drv = bs->drv;
    if (!drv->bdrv_co_pwritev_compressed) {
        return -ENOTSUP;
    return drv->bdrv_co_pwritev_compressed(bs, offset, bytes, qiov);