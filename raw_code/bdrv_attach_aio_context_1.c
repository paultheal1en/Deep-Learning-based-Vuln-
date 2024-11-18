void bdrv_attach_aio_context(BlockDriverState *bs,

                             AioContext *new_context)

{

    BdrvAioNotifier *ban;

    BdrvChild *child;



    if (!bs->drv) {

        return;

    }



    bs->aio_context = new_context;



    QLIST_FOREACH(child, &bs->children, next) {

        bdrv_attach_aio_context(child->bs, new_context);

    }

    if (bs->drv->bdrv_attach_aio_context) {

        bs->drv->bdrv_attach_aio_context(bs, new_context);

    }



    QLIST_FOREACH(ban, &bs->aio_notifiers, list) {

        ban->attached_aio_context(new_context, ban->opaque);

    }

}
