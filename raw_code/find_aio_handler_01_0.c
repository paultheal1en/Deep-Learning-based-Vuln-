static AioHandler *find_aio_handler(AioContext *ctx, int fd)

{

    AioHandler *node;



    QLIST_FOREACH(node, &ctx->aio_handlers, node) {

        if (node->pfd.fd == fd)

            if (!node->deleted)

                return node;

    }



    return NULL;

}
