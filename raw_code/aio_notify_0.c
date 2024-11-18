void aio_notify(AioContext *ctx)

{

    /* Write e.g. bh->scheduled before reading ctx->notify_me.  Pairs

     * with atomic_or in aio_ctx_prepare or atomic_add in aio_poll.

     */

    smp_mb();

    if (ctx->notify_me) {

        event_notifier_set(&ctx->notifier);

        atomic_mb_set(&ctx->notified, true);

    }

}
