void bdrv_io_plug(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (drv && drv->bdrv_io_plug) {

        drv->bdrv_io_plug(bs);

    } else if (bs->file) {

        bdrv_io_plug(bs->file);

    }

}
