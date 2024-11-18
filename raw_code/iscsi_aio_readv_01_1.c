iscsi_aio_readv(BlockDriverState *bs, int64_t sector_num,
                QEMUIOVector *qiov, int nb_sectors,
                BlockDriverCompletionFunc *cb,
                void *opaque)
{
    IscsiLun *iscsilun = bs->opaque;
    IscsiAIOCB *acb;
    acb = qemu_aio_get(&iscsi_aiocb_info, bs, cb, opaque);
    trace_iscsi_aio_readv(iscsilun->iscsi, sector_num, nb_sectors, opaque, acb);
    acb->nb_sectors  = nb_sectors;
    acb->sector_num  = sector_num;
    acb->iscsilun    = iscsilun;
    acb->qiov        = qiov;
    acb->retries     = ISCSI_CMD_RETRIES;
    if (iscsi_aio_readv_acb(acb) != 0) {
        qemu_aio_release(acb);
    iscsi_set_events(iscsilun);
    return &acb->common;