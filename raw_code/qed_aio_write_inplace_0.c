static int coroutine_fn qed_aio_write_inplace(QEDAIOCB *acb, uint64_t offset,

                                              size_t len)

{

    /* Allocate buffer for zero writes */

    if (acb->flags & QED_AIOCB_ZERO) {

        struct iovec *iov = acb->qiov->iov;



        if (!iov->iov_base) {

            iov->iov_base = qemu_try_blockalign(acb->bs, iov->iov_len);

            if (iov->iov_base == NULL) {

                return -ENOMEM;

            }

            memset(iov->iov_base, 0, iov->iov_len);

        }

    }



    /* Calculate the I/O vector */

    acb->cur_cluster = offset;

    qemu_iovec_concat(&acb->cur_qiov, acb->qiov, acb->qiov_offset, len);



    /* Do the actual write */

    return qed_aio_write_main(acb);

}
