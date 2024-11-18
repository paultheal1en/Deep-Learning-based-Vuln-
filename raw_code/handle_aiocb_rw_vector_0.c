static size_t handle_aiocb_rw_vector(struct qemu_paiocb *aiocb)

{

    size_t offset = 0;

    ssize_t len;



    do {

        if (aiocb->aio_type == QEMU_PAIO_WRITE)

            len = qemu_pwritev(aiocb->aio_fildes,

                               aiocb->aio_iov,

                               aiocb->aio_niov,

                               aiocb->aio_offset + offset);

         else

            len = qemu_preadv(aiocb->aio_fildes,

                              aiocb->aio_iov,

                              aiocb->aio_niov,

                              aiocb->aio_offset + offset);

    } while (len == -1 && errno == EINTR);



    if (len == -1)

        return -errno;

    return len;

}
