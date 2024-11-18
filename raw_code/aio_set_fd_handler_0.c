void aio_set_fd_handler(AioContext *ctx,

                        int fd,

                        IOHandler *io_read,

                        IOHandler *io_write,

                        void *opaque)

{

    /* fd is a SOCKET in our case */

    AioHandler *node;



    QLIST_FOREACH(node, &ctx->aio_handlers, node) {

        if (node->pfd.fd == fd && !node->deleted) {

            break;

        }

    }



    /* Are we deleting the fd handler? */

    if (!io_read && !io_write) {

        if (node) {

            /* If the lock is held, just mark the node as deleted */

            if (ctx->walking_handlers) {

                node->deleted = 1;

                node->pfd.revents = 0;

            } else {

                /* Otherwise, delete it for real.  We can't just mark it as

                 * deleted because deleted nodes are only cleaned up after

                 * releasing the walking_handlers lock.

                 */

                QLIST_REMOVE(node, node);

                g_free(node);

            }

        }

    } else {

        HANDLE event;



        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = g_malloc0(sizeof(AioHandler));

            node->pfd.fd = fd;

            QLIST_INSERT_HEAD(&ctx->aio_handlers, node, node);

        }



        node->pfd.events = 0;

        if (node->io_read) {

            node->pfd.events |= G_IO_IN;

        }

        if (node->io_write) {

            node->pfd.events |= G_IO_OUT;

        }



        node->e = &ctx->notifier;



        /* Update handler with latest information */

        node->opaque = opaque;

        node->io_read = io_read;

        node->io_write = io_write;



        event = event_notifier_get_handle(&ctx->notifier);

        WSAEventSelect(node->pfd.fd, event,

                       FD_READ | FD_ACCEPT | FD_CLOSE |

                       FD_CONNECT | FD_WRITE | FD_OOB);

    }



    aio_notify(ctx);

}
