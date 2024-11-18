static BlockAIOCB *read_quorum_children(QuorumAIOCB *acb)

{

    BDRVQuorumState *s = acb->common.bs->opaque;

    int i;



    for (i = 0; i < s->num_children; i++) {

        acb->qcrs[i].buf = qemu_blockalign(s->children[i]->bs, acb->qiov->size);

        qemu_iovec_init(&acb->qcrs[i].qiov, acb->qiov->niov);

        qemu_iovec_clone(&acb->qcrs[i].qiov, acb->qiov, acb->qcrs[i].buf);

    }



    for (i = 0; i < s->num_children; i++) {

        bdrv_aio_readv(s->children[i]->bs, acb->sector_num, &acb->qcrs[i].qiov,

                       acb->nb_sectors, quorum_aio_cb, &acb->qcrs[i]);

    }



    return &acb->common;

}
