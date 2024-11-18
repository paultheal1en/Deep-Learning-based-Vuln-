static AioHandler *find_aio_handler(int fd)

{

    AioHandler *node;



    LIST_FOREACH(node, &aio_handlers, node) {

        if (node->fd == fd)

            return node;

    }



    return NULL;

}
