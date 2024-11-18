static void coroutine_fn bdrv_aio_discard_co_entry(void *opaque)

{

    BlockDriverAIOCBCoroutine *acb = opaque;

    BlockDriverState *bs = acb->common.bs;



    acb->req.error = bdrv_co_discard(bs, acb->req.sector, acb->req.nb_sectors);

    acb->bh = qemu_bh_new(bdrv_co_em_bh, acb);

    qemu_bh_schedule(acb->bh);

}
