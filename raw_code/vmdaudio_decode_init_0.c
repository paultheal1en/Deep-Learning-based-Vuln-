static av_cold int vmdaudio_decode_init(AVCodecContext *avctx)

{

    VmdAudioContext *s = avctx->priv_data;



    if (avctx->channels < 1 || avctx->channels > 2) {

        av_log(avctx, AV_LOG_ERROR, "invalid number of channels\n");

        return AVERROR(EINVAL);

    }

    if (avctx->block_align < 1) {

        av_log(avctx, AV_LOG_ERROR, "invalid block align\n");

        return AVERROR(EINVAL);

    }



    avctx->channel_layout = avctx->channels == 1 ? AV_CH_LAYOUT_MONO :

                                                   AV_CH_LAYOUT_STEREO;



    if (avctx->bits_per_coded_sample == 16)

        avctx->sample_fmt = AV_SAMPLE_FMT_S16;

    else

        avctx->sample_fmt = AV_SAMPLE_FMT_U8;

    s->out_bps = av_get_bytes_per_sample(avctx->sample_fmt);



    s->chunk_size = avctx->block_align + avctx->channels * (s->out_bps == 2);



    avcodec_get_frame_defaults(&s->frame);

    avctx->coded_frame = &s->frame;



    av_log(avctx, AV_LOG_DEBUG, "%d channels, %d bits/sample, "

           "block align = %d, sample rate = %d\n",

           avctx->channels, avctx->bits_per_coded_sample, avctx->block_align,

           avctx->sample_rate);



    return 0;

}
