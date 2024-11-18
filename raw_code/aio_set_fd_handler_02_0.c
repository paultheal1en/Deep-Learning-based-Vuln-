void aio_set_fd_handler(AioContext *ctx,

                        int fd,

                        IOHandler *io_read,

                        IOHandler *io_write,

                        AioFlushHandler *io_flush,

                        void *opaque)

{

    AioHandler *node;



    node = find_aio_handler(ctx, fd);



    /* Are we deleting the fd handler? */

    if (!io_read && !io_write) {

        if (node) {

            /* If the lock is held, just mark the node as deleted */

            if (ctx->walking_handlers)

                node->deleted = 1;

            else {

                /* Otherwise, delete it for real.  We can't just mark it as

                 * deleted because deleted nodes are only cleaned up after

                 * releasing the walking_handlers lock.

                 */

                QLIST_REMOVE(node, node);

                g_free(node);

            }

        }

    } else {

        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = g_malloc0(sizeof(AioHandler));

            node->fd = fd;

            QLIST_INSERT_HEAD(&ctx->aio_handlers, node, node);

        }

        /* Update handler with latest information */

        node->io_read = io_read;

        node->io_write = io_write;

        node->io_flush = io_flush;

        node->opaque = opaque;

    }

}
