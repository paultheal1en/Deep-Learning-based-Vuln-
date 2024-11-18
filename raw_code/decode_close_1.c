static av_cold int decode_close(AVCodecContext *avctx)
{
    IVI5DecContext *ctx = avctx->priv_data;
    ff_ivi_free_buffers(&ctx->planes[0]);
    if (ctx->frame.data[0])
        avctx->release_buffer(avctx, &ctx->frame);
    return 0;
}