void bdrv_debug_event(BlockDriverState *bs, BlkDebugEvent event)

{

    if (!bs || !bs->drv || !bs->drv->bdrv_debug_event) {

        return;

    }



    bs->drv->bdrv_debug_event(bs, event);

}
