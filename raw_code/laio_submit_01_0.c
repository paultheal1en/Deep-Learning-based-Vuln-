BlockAIOCB *laio_submit(BlockDriverState *bs, LinuxAioState *s, int fd,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockCompletionFunc *cb, void *opaque, int type)

{

    struct qemu_laiocb *laiocb;

    struct iocb *iocbs;

    off_t offset = sector_num * 512;



    laiocb = qemu_aio_get(&laio_aiocb_info, bs, cb, opaque);

    laiocb->nbytes = nb_sectors * 512;

    laiocb->ctx = s;

    laiocb->ret = -EINPROGRESS;

    laiocb->is_read = (type == QEMU_AIO_READ);

    laiocb->qiov = qiov;



    iocbs = &laiocb->iocb;



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

        goto out_free_aiocb;

    }

    io_set_eventfd(&laiocb->iocb, event_notifier_get_fd(&s->e));



    QSIMPLEQ_INSERT_TAIL(&s->io_q.pending, laiocb, next);

    s->io_q.n++;

    if (!s->io_q.blocked &&

        (!s->io_q.plugged || s->io_q.n >= MAX_QUEUED_IO)) {

        ioq_submit(s);

    }

    return &laiocb->common;



out_free_aiocb:

    qemu_aio_unref(laiocb);

    return NULL;

}
