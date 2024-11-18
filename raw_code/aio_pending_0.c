bool aio_pending(AioContext *ctx)

{

    AioHandler *node;

    bool result = false;



    /*

     * We have to walk very carefully in case aio_set_fd_handler is

     * called while we're walking.

     */

    qemu_lockcnt_inc(&ctx->list_lock);

    QLIST_FOREACH_RCU(node, &ctx->aio_handlers, node) {

        if (node->pfd.revents && node->io_notify) {

            result = true;

            break;

        }



        if ((node->pfd.revents & G_IO_IN) && node->io_read) {

            result = true;

            break;

        }

        if ((node->pfd.revents & G_IO_OUT) && node->io_write) {

            result = true;

            break;

        }

    }



    qemu_lockcnt_dec(&ctx->list_lock);

    return result;

}
