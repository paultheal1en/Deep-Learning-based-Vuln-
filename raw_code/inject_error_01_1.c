static BlockAIOCB *inject_error(BlockDriverState *bs,

    BlockCompletionFunc *cb, void *opaque, BlkdebugRule *rule)

{

    BDRVBlkdebugState *s = bs->opaque;

    int error = rule->options.inject.error;

    struct BlkdebugAIOCB *acb;

    QEMUBH *bh;



    if (rule->options.inject.once) {

        QSIMPLEQ_INIT(&s->active_rules);

    }



    if (rule->options.inject.immediately) {

        return NULL;

    }



    acb = qemu_aio_get(&blkdebug_aiocb_info, bs, cb, opaque);

    acb->ret = -error;



    bh = aio_bh_new(bdrv_get_aio_context(bs), error_callback_bh, acb);

    acb->bh = bh;

    qemu_bh_schedule(bh);



    return &acb->common;

}
