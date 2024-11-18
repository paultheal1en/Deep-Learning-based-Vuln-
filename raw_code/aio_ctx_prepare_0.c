aio_ctx_prepare(GSource *source, gint    *timeout)

{

    AioContext *ctx = (AioContext *) source;

    uint32_t wait = -1;

    aio_bh_update_timeout(ctx, &wait);



    if (wait != -1) {

        *timeout = MIN(*timeout, wait);

        return wait == 0;

    }



    return false;

}
