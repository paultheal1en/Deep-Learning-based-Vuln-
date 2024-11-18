static av_cold int pcm_decode_init(AVCodecContext *avctx)

{

    PCMDecode *s = avctx->priv_data;

    int i;



    if (avctx->channels <= 0) {

        av_log(avctx, AV_LOG_ERROR, "PCM channels out of bounds\n");

        return AVERROR(EINVAL);

    }



    switch (avctx->codec->id) {

    case AV_CODEC_ID_PCM_ALAW:

        for (i = 0; i < 256; i++)

            s->table[i] = alaw2linear(i);

        break;

    case AV_CODEC_ID_PCM_MULAW:

        for (i = 0; i < 256; i++)

            s->table[i] = ulaw2linear(i);

        break;

    default:

        break;

    }



    avctx->sample_fmt = avctx->codec->sample_fmts[0];



    if (avctx->sample_fmt == AV_SAMPLE_FMT_S32)

        avctx->bits_per_raw_sample = av_get_bits_per_sample(avctx->codec->id);



    avcodec_get_frame_defaults(&s->frame);

    avctx->coded_frame = &s->frame;



    return 0;

}
