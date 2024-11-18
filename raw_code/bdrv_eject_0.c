int bdrv_eject(BlockDriverState *bs, int eject_flag)

{

    BlockDriver *drv = bs->drv;



    if (bs->locked) {

        return -EBUSY;

    }



    if (drv && drv->bdrv_eject) {

        drv->bdrv_eject(bs, eject_flag);

    }

    bs->tray_open = eject_flag;

    return 0;

}
