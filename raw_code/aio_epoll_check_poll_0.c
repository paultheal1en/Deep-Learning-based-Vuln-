static bool aio_epoll_check_poll(AioContext *ctx, GPollFD *pfds,

                                 unsigned npfd, int64_t timeout)

{

    if (!ctx->epoll_available) {

        return false;

    }

    if (aio_epoll_enabled(ctx)) {

        return true;

    }

    if (npfd >= EPOLL_ENABLE_THRESHOLD) {

        if (aio_epoll_try_enable(ctx)) {

            return true;

        } else {

            aio_epoll_disable(ctx);

        }

    }

    return false;

}
