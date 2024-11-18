void qemu_bh_schedule(QEMUBH *bh)

{

    AioContext *ctx;



    ctx = bh->ctx;

    bh->idle = 0;

    /* The memory barrier implicit in atomic_xchg makes sure that:

     * 1. idle & any writes needed by the callback are done before the

     *    locations are read in the aio_bh_poll.

     * 2. ctx is loaded before scheduled is set and the callback has a chance

     *    to execute.

     */

    if (atomic_xchg(&bh->scheduled, 1) == 0) {

        aio_notify(ctx);

    }

}
