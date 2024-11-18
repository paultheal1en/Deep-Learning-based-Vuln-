BlockAIOCB *bdrv_aio_ioctl(BlockDriverState *bs,

        unsigned long int req, void *buf,

        BlockCompletionFunc *cb, void *opaque)

{

    BlockAIOCBCoroutine *acb = qemu_aio_get(&bdrv_em_co_aiocb_info,

                                            bs, cb, opaque);

    Coroutine *co;



    acb->need_bh = true;

    acb->req.error = -EINPROGRESS;

    acb->req.req = req;

    acb->req.buf = buf;

    co = qemu_coroutine_create(bdrv_co_aio_ioctl_entry);

    qemu_coroutine_enter(co, acb);



    bdrv_co_maybe_schedule_bh(acb);

    return &acb->common;

}
