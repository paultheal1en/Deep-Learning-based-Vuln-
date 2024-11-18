static av_cold int alac_encode_close(AVCodecContext *avctx)

{

    AlacEncodeContext *s = avctx->priv_data;

    ff_lpc_end(&s->lpc_ctx);

    av_freep(&avctx->extradata);

    avctx->extradata_size = 0;

    av_freep(&avctx->coded_frame);

    return 0;

}
