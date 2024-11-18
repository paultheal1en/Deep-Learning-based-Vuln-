bool qemu_aio_wait(void)

{

    AioHandler *node;

    fd_set rdfds, wrfds;

    int max_fd = -1;

    int ret;

    bool busy;



    /*

     * If there are callbacks left that have been queued, we need to call then.

     * Do not call select in this case, because it is possible that the caller

     * does not need a complete flush (as is the case for qemu_aio_wait loops).

     */

    if (qemu_bh_poll()) {

        return true;

    }



    walking_handlers++;



    FD_ZERO(&rdfds);

    FD_ZERO(&wrfds);



    /* fill fd sets */

    busy = false;

    QLIST_FOREACH(node, &aio_handlers, node) {

        /* If there aren't pending AIO operations, don't invoke callbacks.

         * Otherwise, if there are no AIO requests, qemu_aio_wait() would

         * wait indefinitely.

         */

        if (node->io_flush) {

            if (node->io_flush(node->opaque) == 0) {

                continue;

            }

            busy = true;

        }

        if (!node->deleted && node->io_read) {

            FD_SET(node->fd, &rdfds);

            max_fd = MAX(max_fd, node->fd + 1);

        }

        if (!node->deleted && node->io_write) {

            FD_SET(node->fd, &wrfds);

            max_fd = MAX(max_fd, node->fd + 1);

        }

    }



    walking_handlers--;



    /* No AIO operations?  Get us out of here */

    if (!busy) {

        return false;

    }



    /* wait until next event */

    ret = select(max_fd, &rdfds, &wrfds, NULL, NULL);



    /* if we have any readable fds, dispatch event */

    if (ret > 0) {

        walking_handlers++;



        /* we have to walk very carefully in case

         * qemu_aio_set_fd_handler is called while we're walking */

        node = QLIST_FIRST(&aio_handlers);

        while (node) {

            AioHandler *tmp;



            if (!node->deleted &&

                FD_ISSET(node->fd, &rdfds) &&

                node->io_read) {

                node->io_read(node->opaque);

            }

            if (!node->deleted &&

                FD_ISSET(node->fd, &wrfds) &&

                node->io_write) {

                node->io_write(node->opaque);

            }



            tmp = node;

            node = QLIST_NEXT(node, node);



            if (tmp->deleted) {

                QLIST_REMOVE(tmp, node);

                g_free(tmp);

            }

        }



        walking_handlers--;

    }



    return true;

}
