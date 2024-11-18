static size_t handle_aiocb_rw_linear(struct qemu_paiocb *aiocb, char *buf)

{

    size_t offset = 0;

    size_t len;



    while (offset < aiocb->aio_nbytes) {

         if (aiocb->aio_type == QEMU_PAIO_WRITE)

             len = pwrite(aiocb->aio_fildes,

                          (const char *)buf + offset,

                          aiocb->aio_nbytes - offset,

                          aiocb->aio_offset + offset);

         else

             len = pread(aiocb->aio_fildes,

                         buf + offset,

                         aiocb->aio_nbytes - offset,

                         aiocb->aio_offset + offset);



         if (len == -1 && errno == EINTR)

             continue;

         else if (len == -1) {

             offset = -errno;

             break;

         } else if (len == 0)

             break;



         offset += len;

    }



    return offset;

}
