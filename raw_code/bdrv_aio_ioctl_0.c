BlockAIOCB *bdrv_aio_ioctl(BlockDriverState *bs,

        unsigned long int req, void *buf,

        BlockCompletionFunc *cb, void *opaque)

{

    BlockDriver *drv = bs->drv;



    if (drv && drv->bdrv_aio_ioctl)

        return drv->bdrv_aio_ioctl(bs, req, buf, cb, opaque);

    return NULL;

}
