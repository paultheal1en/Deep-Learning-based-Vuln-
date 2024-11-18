static RawAIOCB *raw_aio_setup(BlockDriverState *bs, int64_t sector_num,

        QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    BDRVRawState *s = bs->opaque;

    RawAIOCB *acb;



    if (fd_open(bs) < 0)

        return NULL;



    acb = qemu_aio_get(&raw_aio_pool, bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->aiocb.aio_fildes = s->fd;

    acb->aiocb.ev_signo = SIGUSR2;

    acb->aiocb.aio_iov = qiov->iov;

    acb->aiocb.aio_niov = qiov->niov;

    acb->aiocb.aio_nbytes = nb_sectors * 512;

    acb->aiocb.aio_offset = sector_num * 512;

    acb->aiocb.aio_flags = 0;



    /*

     * If O_DIRECT is used the buffer needs to be aligned on a sector

     * boundary. Tell the low level code to ensure that in case it's

     * not done yet.

     */

    if (s->aligned_buf)

        acb->aiocb.aio_flags |= QEMU_AIO_SECTOR_ALIGNED;



    acb->next = posix_aio_state->first_aio;

    posix_aio_state->first_aio = acb;

    return acb;

}
