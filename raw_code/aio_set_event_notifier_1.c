void aio_set_event_notifier(AioContext *ctx,

                            EventNotifier *e,

                            bool is_external,

                            EventNotifierHandler *io_notify)

{

    AioHandler *node;



    QLIST_FOREACH(node, &ctx->aio_handlers, node) {

        if (node->e == e && !node->deleted) {

            break;

        }

    }



    /* Are we deleting the fd handler? */

    if (!io_notify) {

        if (node) {

            g_source_remove_poll(&ctx->source, &node->pfd);



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

        if (node == NULL) {

            /* Alloc and insert if it's not already there */

            node = g_new0(AioHandler, 1);

            node->e = e;

            node->pfd.fd = (uintptr_t)event_notifier_get_handle(e);

            node->pfd.events = G_IO_IN;

            node->is_external = is_external;

            QLIST_INSERT_HEAD(&ctx->aio_handlers, node, node);



            g_source_add_poll(&ctx->source, &node->pfd);

        }

        /* Update handler with latest information */

        node->io_notify = io_notify;

    }



    aio_notify(ctx);

}
