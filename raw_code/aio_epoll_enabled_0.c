static bool aio_epoll_enabled(AioContext *ctx)

{

    /* Fall back to ppoll when external clients are disabled. */

    return !aio_external_disabled(ctx) && ctx->epoll_enabled;

}
