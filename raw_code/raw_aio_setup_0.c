static RawAIOCB *raw_aio_setup(BlockDriverState *bs,

        int64_t sector_num, uint8_t *buf, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    BDRVRawState *s = bs->opaque;

    RawAIOCB *acb;



    if (fd_open(bs) < 0)

        return NULL;



    acb = qemu_aio_get(bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->fd = raw_fd_pool_get(s);

    acb->aiocb.aio_fildes = acb->fd;

    acb->aiocb.aio_sigevent.sigev_signo = SIGUSR2;

    acb->aiocb.aio_sigevent.sigev_notify = SIGEV_SIGNAL;

    acb->aiocb.aio_buf = buf;

    if (nb_sectors < 0)

        acb->aiocb.aio_nbytes = -nb_sectors;

    else

        acb->aiocb.aio_nbytes = nb_sectors * 512;

    acb->aiocb.aio_offset = sector_num * 512;

    acb->next = posix_aio_state->first_aio;

    posix_aio_state->first_aio = acb;

    return acb;

}
