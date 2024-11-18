void aio_set_dispatching(AioContext *ctx, bool dispatching)

{

    ctx->dispatching = dispatching;

    if (!dispatching) {

        /* Write ctx->dispatching before reading e.g. bh->scheduled.

         * Optimization: this is only needed when we're entering the "unsafe"

         * phase where other threads must call event_notifier_set.

         */

        smp_mb();

    }

}
