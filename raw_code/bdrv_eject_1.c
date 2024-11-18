void bdrv_eject(BlockDriverState *bs, int eject_flag)

{

    BlockDriver *drv = bs->drv;

    int ret;



    if (!drv || !drv->bdrv_eject) {

        ret = -ENOTSUP;

    } else {

        ret = drv->bdrv_eject(bs, eject_flag);

    }

    if (ret == -ENOTSUP) {

        if (eject_flag)

            bdrv_close(bs);

    }

}
