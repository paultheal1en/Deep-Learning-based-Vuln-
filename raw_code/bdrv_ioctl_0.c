int bdrv_ioctl(BlockDriverState *bs, unsigned long int req, void *buf)

{

    BlockDriver *drv = bs->drv;



    if (drv && drv->bdrv_ioctl)

        return drv->bdrv_ioctl(bs, req, buf);

    return -ENOTSUP;

}
