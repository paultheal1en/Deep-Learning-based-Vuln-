static av_cold int bmv_aud_decode_init(AVCodecContext *avctx)

{

    BMVAudioDecContext *c = avctx->priv_data;



    if (avctx->channels != 2) {

        av_log(avctx, AV_LOG_INFO, "invalid number of channels\n");

        return AVERROR(EINVAL);

    }



    avctx->sample_fmt = AV_SAMPLE_FMT_S16;



    avcodec_get_frame_defaults(&c->frame);

    avctx->coded_frame = &c->frame;



    return 0;

}
