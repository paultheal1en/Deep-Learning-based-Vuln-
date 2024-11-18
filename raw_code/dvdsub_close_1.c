static av_cold int dvdsub_close(AVCodecContext *avctx)

{

    DVDSubContext *ctx = avctx->priv_data;

    av_freep(&ctx->buf);

    ctx->buf_size = 0;

    return 0;

}
