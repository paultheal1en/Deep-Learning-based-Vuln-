void aio_notify(AioContext *ctx)

{

    /* Write e.g. bh->scheduled before reading ctx->dispatching.  */

    smp_mb();

    if (!ctx->dispatching) {

        event_notifier_set(&ctx->notifier);

    }

}
