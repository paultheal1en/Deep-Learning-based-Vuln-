ThreadPool *aio_get_thread_pool(AioContext *ctx)

{

    if (!ctx->thread_pool) {

        ctx->thread_pool = thread_pool_new(ctx);

    }

    return ctx->thread_pool;

}
