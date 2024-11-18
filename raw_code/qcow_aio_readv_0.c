static BlockDriverAIOCB *qcow_aio_readv(BlockDriverState *bs,

        int64_t sector_num, QEMUIOVector *qiov, int nb_sectors,

        BlockDriverCompletionFunc *cb, void *opaque)

{

    QCowAIOCB *acb;



    acb = qemu_aio_get(bs, cb, opaque);

    if (!acb)

        return NULL;

    acb->hd_aiocb = NULL;

    acb->sector_num = sector_num;

    acb->qiov = qiov;

    if (qiov->niov > 1)

        acb->buf = acb->orig_buf = qemu_memalign(512, qiov->size);

    else

        acb->buf = (uint8_t *)qiov->iov->iov_base;

    acb->nb_sectors = nb_sectors;

    acb->n = 0;

    acb->cluster_offset = 0;



    qcow_aio_read_cb(acb, 0);

    return &acb->common;

}
