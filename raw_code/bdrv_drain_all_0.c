void bdrv_drain_all(void)

{

    /* Always run first iteration so any pending completion BHs run */

    bool busy = true;

    BlockDriverState *bs = NULL;

    GSList *aio_ctxs = NULL, *ctx;



    while ((bs = bdrv_next(bs))) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        if (bs->job) {

            block_job_pause(bs->job);

        }

        bdrv_no_throttling_begin(bs);

        bdrv_drain_recurse(bs);

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

    while (busy) {

        busy = false;



        for (ctx = aio_ctxs; ctx != NULL; ctx = ctx->next) {

            AioContext *aio_context = ctx->data;

            bs = NULL;



            aio_context_acquire(aio_context);

            while ((bs = bdrv_next(bs))) {

                if (aio_context == bdrv_get_aio_context(bs)) {

                    bdrv_flush_io_queue(bs);

                    if (bdrv_requests_pending(bs)) {

                        busy = true;

                        aio_poll(aio_context, busy);

                    }

                }

            }

            busy |= aio_poll(aio_context, false);

            aio_context_release(aio_context);

        }

    }



    bs = NULL;

    while ((bs = bdrv_next(bs))) {

        AioContext *aio_context = bdrv_get_aio_context(bs);



        aio_context_acquire(aio_context);

        bdrv_no_throttling_end(bs);

        if (bs->job) {

            block_job_resume(bs->job);

        }

        aio_context_release(aio_context);

    }

    g_slist_free(aio_ctxs);

}
