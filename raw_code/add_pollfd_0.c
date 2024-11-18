static void add_pollfd(AioHandler *node)

{

    if (npfd == nalloc) {

        if (nalloc == 0) {

            pollfds_cleanup_notifier.notify = pollfds_cleanup;

            qemu_thread_atexit_add(&pollfds_cleanup_notifier);

            nalloc = 8;

        } else {

            g_assert(nalloc <= INT_MAX);

            nalloc *= 2;

        }

        pollfds = g_renew(GPollFD, pollfds, nalloc);

        nodes = g_renew(AioHandler *, nodes, nalloc);

    }

    nodes[npfd] = node;

    pollfds[npfd] = (GPollFD) {

        .fd = node->pfd.fd,

        .events = node->pfd.events,

    };

    npfd++;

}
