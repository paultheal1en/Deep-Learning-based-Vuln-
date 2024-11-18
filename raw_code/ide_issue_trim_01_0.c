BlockAIOCB *ide_issue_trim(

        int64_t offset, QEMUIOVector *qiov,

        BlockCompletionFunc *cb, void *cb_opaque, void *opaque)

{

    BlockBackend *blk = opaque;

    TrimAIOCB *iocb;



    iocb = blk_aio_get(&trim_aiocb_info, blk, cb, cb_opaque);

    iocb->blk = blk;

    iocb->bh = qemu_bh_new(ide_trim_bh_cb, iocb);

    iocb->ret = 0;

    iocb->qiov = qiov;

    iocb->i = -1;

    iocb->j = 0;

    ide_issue_trim_cb(iocb, 0);

    return &iocb->common;

}
