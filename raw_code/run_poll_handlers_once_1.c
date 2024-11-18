static bool run_poll_handlers_once(AioContext *ctx)

{

    bool progress = false;

    AioHandler *node;



    QLIST_FOREACH_RCU(node, &ctx->aio_handlers, node) {

        if (!node->deleted && node->io_poll &&

                node->io_poll(node->opaque)) {

            progress = true;

        }



        /* Caller handles freeing deleted nodes.  Don't do it here. */

    }



    return progress;

}
