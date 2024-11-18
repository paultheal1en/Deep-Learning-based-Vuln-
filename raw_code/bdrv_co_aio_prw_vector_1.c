static BlockAIOCB *bdrv_co_aio_prw_vector(BdrvChild *child,

                                          int64_t offset,

                                          QEMUIOVector *qiov,

                                          BdrvRequestFlags flags,

                                          BlockCompletionFunc *cb,

                                          void *opaque,

                                          bool is_write)

{

    Coroutine *co;

    BlockAIOCBCoroutine *acb;



    /* Matched by bdrv_co_complete's bdrv_dec_in_flight.  */

    bdrv_inc_in_flight(child->bs);



    acb = qemu_aio_get(&bdrv_em_co_aiocb_info, child->bs, cb, opaque);

    acb->child = child;

    acb->need_bh = true;

    acb->req.error = -EINPROGRESS;

    acb->req.offset = offset;

    acb->req.qiov = qiov;

    acb->req.flags = flags;

    acb->is_write = is_write;



    co = qemu_coroutine_create(bdrv_co_do_rw, acb);

    qemu_coroutine_enter(co);



    bdrv_co_maybe_schedule_bh(acb);

    return &acb->common;

}
