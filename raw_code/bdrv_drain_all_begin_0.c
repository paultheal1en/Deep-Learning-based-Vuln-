void bdrv_drain_all_begin(void)

{

    /* Always run first iteration so any pending completion BHs run */

    bool waited = true;

    BlockDriverState *bs;

    BdrvNextIterator it;

    GSList *aio_ctxs = NULL, *ctx;



    block_job_pause_all();



    for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        bdrv_parent_drained_begin(bs);

        aio_disable_external(aio_context);

        aio_context_release(aio_context);



        if (!g_slist_find(aio_ctxs, aio_context)) {

            aio_ctxs = g_slist_prepend(aio_ctxs, aio_context);

        }

    }



    /* Note that completion of an asynchronous I/O operation can trigger any

     * number of other I/O operations on other devices---for example a

     * coroutine can submit an I/O request to another device in response to

     * request completion.  Therefore we must keep looping until there was no

     * more activity rather than simply draining each device independently.

     */

    while (waited) {

        waited = false;



        for (ctx = aio_ctxs; ctx != NULL; ctx = ctx->next) {

            AioContext *aio_context = ctx->data;



            aio_context_acquire(aio_context);

            for (bs = bdrv_first(&it); bs; bs = bdrv_next(&it)) {

                if (aio_context == bdrv_get_aio_context(bs)) {

                    /* FIXME Calling this multiple times is wrong */

                    bdrv_drain_invoke(bs, true);

                    waited |= bdrv_drain_recurse(bs, true);

                }

            }

            aio_context_release(aio_context);

        }

    }



    g_slist_free(aio_ctxs);

}
