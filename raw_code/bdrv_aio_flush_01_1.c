BlockDriverAIOCB *bdrv_aio_flush(BlockDriverState *bs,
        BlockDriverCompletionFunc *cb, void *opaque)
{
    BlockDriver *drv = bs->drv;
    if (!drv)
        return NULL;
    return drv->bdrv_aio_flush(bs, cb, opaque);