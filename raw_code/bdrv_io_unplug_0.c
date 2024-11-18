void bdrv_io_unplug(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (drv && drv->bdrv_io_unplug) {

        drv->bdrv_io_unplug(bs);

    } else if (bs->file) {

        bdrv_io_unplug(bs->file->bs);

    }

}
