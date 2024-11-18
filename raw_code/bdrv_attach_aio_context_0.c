void bdrv_attach_aio_context(BlockDriverState *bs,

                             AioContext *new_context)

{

    BdrvAioNotifier *ban;



    if (!bs->drv) {

        return;

    }



    bs->aio_context = new_context;



    if (bs->backing) {

        bdrv_attach_aio_context(bs->backing->bs, new_context);

    }

    if (bs->file) {

        bdrv_attach_aio_context(bs->file->bs, new_context);

    }

    if (bs->drv->bdrv_attach_aio_context) {

        bs->drv->bdrv_attach_aio_context(bs, new_context);

    }

    if (bs->io_limits_enabled) {

        throttle_timers_attach_aio_context(&bs->throttle_timers, new_context);

    }



    QLIST_FOREACH(ban, &bs->aio_notifiers, list) {

        ban->attached_aio_context(new_context, ban->opaque);

    }

}
