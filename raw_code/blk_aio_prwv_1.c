static BlockAIOCB *blk_aio_prwv(BlockBackend *blk, int64_t offset, int bytes,

                                QEMUIOVector *qiov, CoroutineEntry co_entry,

                                BdrvRequestFlags flags,

                                BlockCompletionFunc *cb, void *opaque)

{

    BlkAioEmAIOCB *acb;

    Coroutine *co;



    bdrv_inc_in_flight(blk_bs(blk));

    acb = blk_aio_get(&blk_aio_em_aiocb_info, blk, cb, opaque);

    acb->rwco = (BlkRwCo) {

        .blk    = blk,

        .offset = offset,

        .qiov   = qiov,

        .flags  = flags,

        .ret    = NOT_DONE,

    };

    acb->bytes = bytes;

    acb->has_returned = false;



    co = qemu_coroutine_create(co_entry, acb);

    qemu_coroutine_enter(co);



    acb->has_returned = true;

    if (acb->rwco.ret != NOT_DONE) {

        aio_bh_schedule_oneshot(blk_get_aio_context(blk),

                                blk_aio_complete_bh, acb);

    }



    return &acb->common;

}
