void aio_bh_schedule_oneshot(AioContext *ctx, QEMUBHFunc *cb, void *opaque)

{

    QEMUBH *bh;

    bh = g_new(QEMUBH, 1);

    *bh = (QEMUBH){

        .ctx = ctx,

        .cb = cb,

        .opaque = opaque,

    };

    qemu_lockcnt_lock(&ctx->list_lock);

    bh->next = ctx->first_bh;

    bh->scheduled = 1;

    bh->deleted = 1;

    /* Make sure that the members are ready before putting bh into list */

    smp_wmb();

    ctx->first_bh = bh;

    qemu_lockcnt_unlock(&ctx->list_lock);

    aio_notify(ctx);

}
