static void aio_epoll_disable(AioContext *ctx)

{

    ctx->epoll_available = false;

    if (!ctx->epoll_enabled) {

        return;

    }

    ctx->epoll_enabled = false;

    close(ctx->epollfd);

}
