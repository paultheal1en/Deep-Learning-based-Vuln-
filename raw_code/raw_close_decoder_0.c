static av_cold int raw_close_decoder(AVCodecContext *avctx)

{

    RawVideoContext *context = avctx->priv_data;



    av_freep(&context->buffer);

    return 0;

}
