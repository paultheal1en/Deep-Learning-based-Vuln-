GSource *aio_get_g_source(AioContext *ctx)

{

    g_source_ref(&ctx->source);

    return &ctx->source;

}
