void bdrv_flush_io_queue(BlockDriverState *bs)

{

    BlockDriver *drv = bs->drv;

    if (drv && drv->bdrv_flush_io_queue) {

        drv->bdrv_flush_io_queue(bs);

    } else if (bs->file) {

        bdrv_flush_io_queue(bs->file);

    }

}
