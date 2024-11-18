void bdrv_detach_aio_context(BlockDriverState *bs)

{

    BdrvAioNotifier *baf;

    BdrvChild *child;



    if (!bs->drv) {

        return;

    }



    QLIST_FOREACH(baf, &bs->aio_notifiers, list) {

        baf->detach_aio_context(baf->opaque);

    }



    if (bs->drv->bdrv_detach_aio_context) {

        bs->drv->bdrv_detach_aio_context(bs);

    }

    QLIST_FOREACH(child, &bs->children, next) {

        bdrv_detach_aio_context(child->bs);

    }



    bs->aio_context = NULL;

}
