static BlockDriverAIOCB *paio_submit(BlockDriverState *bs, HANDLE hfile,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque, int type)

{

    RawWin32AIOData *acb = g_slice_new(RawWin32AIOData);



    acb->bs = bs;

    acb->hfile = hfile;

    acb->aio_type = type;



    if (qiov) {

        acb->aio_iov = qiov->iov;

        acb->aio_niov = qiov->niov;

    }

    acb->aio_nbytes = nb_sectors * 512;

    acb->aio_offset = sector_num * 512;



    trace_paio_submit(acb, opaque, sector_num, nb_sectors, type);

    return thread_pool_submit_aio(aio_worker, acb, cb, opaque);

}
