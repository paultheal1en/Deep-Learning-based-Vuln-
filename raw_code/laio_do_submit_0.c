static int laio_do_submit(int fd, struct qemu_laiocb *laiocb, off_t offset,

                          int type)

{

    LinuxAioState *s = laiocb->ctx;

    struct iocb *iocbs = &laiocb->iocb;

    QEMUIOVector *qiov = laiocb->qiov;



    switch (type) {

    case QEMU_AIO_WRITE:

        io_prep_pwritev(iocbs, fd, qiov->iov, qiov->niov, offset);

	break;

    case QEMU_AIO_READ:

        io_prep_preadv(iocbs, fd, qiov->iov, qiov->niov, offset);

	break;

    /* Currently Linux kernel does not support other operations */

    default:

        fprintf(stderr, "%s: invalid AIO request type 0x%x.\n",

                        __func__, type);

        return -EIO;

    }

    io_set_eventfd(&laiocb->iocb, event_notifier_get_fd(&s->e));



    QSIMPLEQ_INSERT_TAIL(&s->io_q.pending, laiocb, next);

    s->io_q.n++;

    if (!s->io_q.blocked &&

        (!s->io_q.plugged || s->io_q.n >= MAX_QUEUED_IO)) {

        ioq_submit(s);

    }



    return 0;

}
