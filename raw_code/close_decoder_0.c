static av_cold int close_decoder(AVCodecContext *avctx)

{

    PGSSubContext *ctx = avctx->priv_data;



    av_freep(&ctx->picture.rle);

    ctx->picture.rle_buffer_size  = 0;



    return 0;

}
