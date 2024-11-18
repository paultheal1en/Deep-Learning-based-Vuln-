static void coroutine_fn bdrv_aio_flush_co_entry(void *opaque)

{

    BlockAIOCBCoroutine *acb = opaque;

    BlockDriverState *bs = acb->common.bs;



    acb->req.error = bdrv_co_flush(bs);

    acb->bh = aio_bh_new(bdrv_get_aio_context(bs), bdrv_co_em_bh, acb);

    qemu_bh_schedule(acb->bh);

}
