static bool event_notifier_poll(void *opaque)

{

    EventNotifier *e = opaque;

    AioContext *ctx = container_of(e, AioContext, notifier);



    return atomic_read(&ctx->notified);

}
