static BlockDriverAIOCB *quorum_aio_readv(BlockDriverState *bs,

                                         int64_t sector_num,

                                         QEMUIOVector *qiov,

                                         int nb_sectors,

                                         BlockDriverCompletionFunc *cb,

                                         void *opaque)

{

    BDRVQuorumState *s = bs->opaque;

    QuorumAIOCB *acb = quorum_aio_get(s, bs, qiov, sector_num,

                                      nb_sectors, cb, opaque);

    int i;



    acb->is_read = true;



    for (i = 0; i < s->num_children; i++) {

        acb->qcrs[i].buf = qemu_blockalign(s->bs[i], qiov->size);

        qemu_iovec_init(&acb->qcrs[i].qiov, qiov->niov);

        qemu_iovec_clone(&acb->qcrs[i].qiov, qiov, acb->qcrs[i].buf);

    }



    for (i = 0; i < s->num_children; i++) {

        bdrv_aio_readv(s->bs[i], sector_num, &acb->qcrs[i].qiov, nb_sectors,

                       quorum_aio_cb, &acb->qcrs[i]);

    }



    return &acb->common;

}
