bool aio_poll(AioContext *ctx, bool blocking)

{

    AioHandler *node;

    int i, ret;

    bool progress;

    int64_t timeout;



    aio_context_acquire(ctx);

    progress = false;



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



    ctx->walking_handlers++;



    assert(npfd == 0);



    /* fill pollfds */



    if (!aio_epoll_enabled(ctx)) {

        QLIST_FOREACH(node, &ctx->aio_handlers, node) {

            if (!node->deleted && node->pfd.events

                && aio_node_check(ctx, node->is_external)) {

                add_pollfd(node);

            }

        }

    }



    timeout = blocking ? aio_compute_timeout(ctx) : 0;



    /* wait until next event */

    if (timeout) {

        aio_context_release(ctx);

    }

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

    if (blocking) {

        atomic_sub(&ctx->notify_me, 2);

    }

    if (timeout) {

        aio_context_acquire(ctx);

    }



    aio_notify_accept(ctx);



    /* if we have any readable fds, dispatch event */

    if (ret > 0) {

        for (i = 0; i < npfd; i++) {

            nodes[i]->pfd.revents = pollfds[i].revents;

        }

    }



    npfd = 0;

    ctx->walking_handlers--;



    /* Run dispatch even if there were no readable fds to run timers */

    if (aio_dispatch(ctx, ret > 0)) {

        progress = true;

    }



    aio_context_release(ctx);



    return progress;

}
