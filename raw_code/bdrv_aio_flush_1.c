BlockAIOCB *bdrv_aio_flush(BlockDriverState *bs,

        BlockCompletionFunc *cb, void *opaque)

{

    trace_bdrv_aio_flush(bs, opaque);



    Coroutine *co;

    BlockAIOCBCoroutine *acb;



    /* Matched by bdrv_co_complete's bdrv_dec_in_flight.  */

    bdrv_inc_in_flight(bs);



    acb = qemu_aio_get(&bdrv_em_co_aiocb_info, bs, cb, opaque);

    acb->need_bh = true;

    acb->req.error = -EINPROGRESS;



    co = qemu_coroutine_create(bdrv_aio_flush_co_entry, acb);

    qemu_coroutine_enter(co);



    bdrv_co_maybe_schedule_bh(acb);

    return &acb->common;

}
