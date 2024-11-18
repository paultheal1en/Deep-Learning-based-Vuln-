void bdrv_io_unplugged_begin(BlockDriverState *bs)

{

    BdrvChild *child;



    if (bs->io_plug_disabled++ == 0 && bs->io_plugged > 0) {

        BlockDriver *drv = bs->drv;

        if (drv && drv->bdrv_io_unplug) {

            drv->bdrv_io_unplug(bs);

        }

    }



    QLIST_FOREACH(child, &bs->children, next) {

        bdrv_io_unplugged_begin(child->bs);

    }

}
