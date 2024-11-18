static bool aio_dispatch(AioContext *ctx)

{

    AioHandler *node;

    bool progress = false;



    /*

     * We have to walk very carefully in case qemu_aio_set_fd_handler is

     * called while we're walking.

     */

    node = QLIST_FIRST(&ctx->aio_handlers);

    while (node) {

        AioHandler *tmp;

        int revents;



        ctx->walking_handlers++;



        revents = node->pfd.revents & node->pfd.events;

        node->pfd.revents = 0;



        if (!node->deleted &&

            (revents & (G_IO_IN | G_IO_HUP | G_IO_ERR)) &&

            node->io_read) {

            node->io_read(node->opaque);

            progress = true;

        }

        if (!node->deleted &&

            (revents & (G_IO_OUT | G_IO_ERR)) &&

            node->io_write) {

            node->io_write(node->opaque);

            progress = true;

        }



        tmp = node;

        node = QLIST_NEXT(node, node);



        ctx->walking_handlers--;



        if (!ctx->walking_handlers && tmp->deleted) {

            QLIST_REMOVE(tmp, node);

            g_free(tmp);

        }

    }

    return progress;

}
