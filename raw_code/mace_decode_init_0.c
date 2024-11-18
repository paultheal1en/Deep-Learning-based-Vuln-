static av_cold int mace_decode_init(AVCodecContext * avctx)

{

    MACEContext *ctx = avctx->priv_data;



    if (avctx->channels > 2)

        return -1;

    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    avcodec_get_frame_defaults(&ctx->frame);

    avctx->coded_frame = &ctx->frame;



    return 0;

}
