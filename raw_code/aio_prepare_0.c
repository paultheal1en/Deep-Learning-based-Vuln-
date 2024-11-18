bool aio_prepare(AioContext *ctx)

{

    /* Poll mode cannot be used with glib's event loop, disable it. */

    poll_set_started(ctx, false);



    return false;

}
