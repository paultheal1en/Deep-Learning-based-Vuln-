static void laio_cancel(BlockDriverAIOCB *blockacb)

{

    struct qemu_laiocb *laiocb = (struct qemu_laiocb *)blockacb;

    struct io_event event;

    int ret;



    if (laiocb->ret != -EINPROGRESS)

        return;



    /*

     * Note that as of Linux 2.6.31 neither the block device code nor any

     * filesystem implements cancellation of AIO request.

     * Thus the polling loop below is the normal code path.

     */

    ret = io_cancel(laiocb->ctx->ctx, &laiocb->iocb, &event);

    if (ret == 0) {

        laiocb->ret = -ECANCELED;

        return;

    }



    /*

     * We have to wait for the iocb to finish.

     *

     * The only way to get the iocb status update is by polling the io context.

     * We might be able to do this slightly more optimal by removing the

     * O_NONBLOCK flag.

     */

    while (laiocb->ret == -EINPROGRESS) {

        qemu_laio_completion_cb(&laiocb->ctx->e);

    }

}
