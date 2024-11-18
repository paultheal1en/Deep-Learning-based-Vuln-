int coroutine_fn laio_co_submit(BlockDriverState *bs, LinuxAioState *s, int fd,

                                uint64_t offset, QEMUIOVector *qiov, int type)

{

    int ret;

    struct qemu_laiocb laiocb = {

        .co         = qemu_coroutine_self(),

        .nbytes     = qiov->size,

        .ctx        = s,

        .is_read    = (type == QEMU_AIO_READ),

        .qiov       = qiov,

    };



    ret = laio_do_submit(fd, &laiocb, offset, type);

    if (ret < 0) {

        return ret;

    }



    qemu_coroutine_yield();

    return laiocb.ret;

}
