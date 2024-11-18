void bdrv_drain_all(void)

{

    /* Always run first iteration so any pending completion BHs run */

    bool busy = true;

    BlockDriverState *bs;



    while (busy) {

        /* FIXME: We do not have timer support here, so this is effectively

         * a busy wait.

         */

        QTAILQ_FOREACH(bs, &bdrv_states, list) {

            if (bdrv_start_throttled_reqs(bs)) {

                busy = true;

            }

        }



        busy = bdrv_requests_pending_all();

        busy |= aio_poll(qemu_get_aio_context(), busy);

    }

}
