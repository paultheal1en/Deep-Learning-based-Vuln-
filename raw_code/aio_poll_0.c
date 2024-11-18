bool aio_poll(AioContext *ctx, bool blocking)

{

    AioHandler *node;

    int i;

    int ret = 0;

    bool progress;

    int64_t timeout;

    int64_t start = 0;



    /* aio_notify can avoid the expensive event_notifier_set if

     * everything (file descriptors, bottom halves, timers) will

     * be re-evaluated before the next blocking poll().  This is

     * already true when aio_poll is called with blocking == false;

     * if blocking == true, it is only true after poll() returns,

     * so disable the optimization now.

     */

    if (blocking) {

        atomic_add(&ctx->notify_me, 2);

    }



    qemu_lockcnt_inc(&ctx->list_lock);



    if (ctx->poll_max_ns) {

        start = qemu_clock_get_ns(QEMU_CLOCK_REALTIME);

    }



    progress = try_poll_mode(ctx, blocking);

    if (!progress) {

        assert(npfd == 0);



        /* fill pollfds */



        if (!aio_epoll_enabled(ctx)) {

            QLIST_FOREACH_RCU(node, &ctx->aio_handlers, node) {

                if (!node->deleted && node->pfd.events

                    && aio_node_check(ctx, node->is_external)) {

                    add_pollfd(node);

                }

            }

        }



        timeout = blocking ? aio_compute_timeout(ctx) : 0;



        /* wait until next event */

        if (aio_epoll_check_poll(ctx, pollfds, npfd, timeout)) {

            AioHandler epoll_handler;



            epoll_handler.pfd.fd = ctx->epollfd;

            epoll_handler.pfd.events = G_IO_IN | G_IO_OUT | G_IO_HUP | G_IO_ERR;

            npfd = 0;

            add_pollfd(&epoll_handler);

            ret = aio_epoll(ctx, pollfds, npfd, timeout);

        } else  {

            ret = qemu_poll_ns(pollfds, npfd, timeout);

        }

    }



    if (blocking) {

        atomic_sub(&ctx->notify_me, 2);

    }



    /* Adjust polling time */

    if (ctx->poll_max_ns) {

        int64_t block_ns = qemu_clock_get_ns(QEMU_CLOCK_REALTIME) - start;



        if (block_ns <= ctx->poll_ns) {

            /* This is the sweet spot, no adjustment needed */

        } else if (block_ns > ctx->poll_max_ns) {

            /* We'd have to poll for too long, poll less */

            int64_t old = ctx->poll_ns;



            if (ctx->poll_shrink) {

                ctx->poll_ns /= ctx->poll_shrink;

            } else {

                ctx->poll_ns = 0;

            }



            trace_poll_shrink(ctx, old, ctx->poll_ns);

        } else if (ctx->poll_ns < ctx->poll_max_ns &&

                   block_ns < ctx->poll_max_ns) {

            /* There is room to grow, poll longer */

            int64_t old = ctx->poll_ns;

            int64_t grow = ctx->poll_grow;



            if (grow == 0) {

                grow = 2;

            }



            if (ctx->poll_ns) {

                ctx->poll_ns *= grow;

            } else {

                ctx->poll_ns = 4000; /* start polling at 4 microseconds */

            }



            if (ctx->poll_ns > ctx->poll_max_ns) {

                ctx->poll_ns = ctx->poll_max_ns;

            }



            trace_poll_grow(ctx, old, ctx->poll_ns);

        }

    }



    aio_notify_accept(ctx);



    /* if we have any readable fds, dispatch event */

    if (ret > 0) {

        for (i = 0; i < npfd; i++) {

            nodes[i]->pfd.revents = pollfds[i].revents;

        }

    }



    npfd = 0;

    qemu_lockcnt_dec(&ctx->list_lock);



    /* Run dispatch even if there were no readable fds to run timers */

    if (aio_dispatch(ctx, ret > 0)) {

        progress = true;

    }



    return progress;

}
