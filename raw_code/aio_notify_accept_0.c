void aio_notify_accept(AioContext *ctx)

{

    if (atomic_xchg(&ctx->notified, false)) {

        event_notifier_test_and_clear(&ctx->notifier);

    }

}
