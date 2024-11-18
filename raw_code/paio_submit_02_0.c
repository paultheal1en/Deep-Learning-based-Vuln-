static BlockDriverAIOCB *paio_submit(BlockDriverState *bs, int fd,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    RawPosixAIOData *acb = g_slice_new(RawPosixAIOData);



    acb->bs = bs;

    acb->aio_type = type;

    acb->aio_fildes = fd;



    if (qiov) {

        acb->aio_iov = qiov->iov;

        acb->aio_niov = qiov->niov;

    }

    acb->aio_nbytes = nb_sectors * 512;

    acb->aio_offset = sector_num * 512;



    trace_paio_submit(acb, opaque, sector_num, nb_sectors, type);

    return thread_pool_submit_aio(aio_worker, acb, cb, opaque);

}
