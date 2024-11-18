AioContext *aio_context_new(Error **errp)

{

    int ret;

    AioContext *ctx;



    ctx = (AioContext *) g_source_new(&aio_source_funcs, sizeof(AioContext));

    aio_context_setup(ctx);



    ret = event_notifier_init(&ctx->notifier, false);

    if (ret < 0) {

        error_setg_errno(errp, -ret, "Failed to initialize event notifier");

        goto fail;

    }

    g_source_set_can_recurse(&ctx->source, true);

    aio_set_event_notifier(ctx, &ctx->notifier,

                           false,

                           (EventNotifierHandler *)

                           event_notifier_dummy_cb,

                           event_notifier_poll);

#ifdef CONFIG_LINUX_AIO

    ctx->linux_aio = NULL;

#endif

    ctx->thread_pool = NULL;

    qemu_mutex_init(&ctx->bh_lock);

    qemu_rec_mutex_init(&ctx->lock);

    timerlistgroup_init(&ctx->tlg, aio_timerlist_notify, ctx);




    ctx->poll_max_ns = 0;





    return ctx;

fail:

    g_source_destroy(&ctx->source);

    return NULL;

}