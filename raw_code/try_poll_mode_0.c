static bool try_poll_mode(AioContext *ctx, bool blocking)

{

    if (blocking && ctx->poll_max_ns && ctx->poll_disable_cnt == 0) {

        /* See qemu_soonest_timeout() uint64_t hack */

        int64_t max_ns = MIN((uint64_t)aio_compute_timeout(ctx),

                             (uint64_t)ctx->poll_ns);



        if (max_ns) {

            poll_set_started(ctx, true);



            if (run_poll_handlers(ctx, max_ns)) {

                return true;

            }

        }

    }



    poll_set_started(ctx, false);



    /* Even if we don't run busy polling, try polling once in case it can make

     * progress and the caller will be able to avoid ppoll(2)/epoll_wait(2).

     */

    return run_poll_handlers_once(ctx);

}
