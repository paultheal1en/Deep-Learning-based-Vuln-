int bdrv_writev_vmstate(BlockDriverState *bs, QEMUIOVector *qiov, int64_t pos)

{

    BlockDriver *drv = bs->drv;



    if (!drv) {

        return -ENOMEDIUM;

    } else if (drv->bdrv_save_vmstate) {

        return drv->bdrv_save_vmstate(bs, qiov, pos);

    } else if (bs->file) {

        return bdrv_writev_vmstate(bs->file, qiov, pos);

    }



    return -ENOTSUP;

}
