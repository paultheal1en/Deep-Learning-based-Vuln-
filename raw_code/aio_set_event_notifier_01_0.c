void aio_set_event_notifier(AioContext *ctx,

                            EventNotifier *e,

                            bool is_external,

                            EventNotifierHandler *io_notify,

                            AioPollFn *io_poll)

{

    AioHandler *node;



    qemu_lockcnt_lock(&ctx->list_lock);

    QLIST_FOREACH(node, &ctx->aio_handlers, node) {

        if (node->e == e && !node->deleted) {

            break;

        }

    }



    /* Are we deleting the fd handler? */

    if (!io_notify) {

        if (node) {

            g_source_remove_poll(&ctx->source, &node->pfd);



            /* aio_poll is in progress, just mark the node as deleted */

            if (qemu_lockcnt_count(&ctx->list_lock)) {

                node->deleted = 1;

                node->pfd.revents = 0;

            } else {

                /* Otherwise, delete it for real.  We can't just mark it as

                 * deleted because deleted nodes are only cleaned up after

                 * releasing the list_lock.

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

            QLIST_INSERT_HEAD_RCU(&ctx->aio_handlers, node, node);



            g_source_add_poll(&ctx->source, &node->pfd);

        }

        /* Update handler with latest information */

        node->io_notify = io_notify;

    }



    qemu_lockcnt_unlock(&ctx->list_lock);

    aio_notify(ctx);

}
