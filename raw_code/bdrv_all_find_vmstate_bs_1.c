BlockDriverState *bdrv_all_find_vmstate_bs(void)

{

    bool not_found = true;

    BlockDriverState *bs;

    BdrvNextIterator *it = NULL;



    while (not_found && (it = bdrv_next(it, &bs))) {

        AioContext *ctx = bdrv_get_aio_context(bs);



        aio_context_acquire(ctx);

        not_found = !bdrv_can_snapshot(bs);

        aio_context_release(ctx);

    }

    return bs;

}
