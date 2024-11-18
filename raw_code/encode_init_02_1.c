static av_cold int encode_init(AVCodecContext *avctx)

{

    NellyMoserEncodeContext *s = avctx->priv_data;

    int i, ret;



    if (avctx->channels != 1) {

        av_log(avctx, AV_LOG_ERROR, "Nellymoser supports only 1 channel\n");

        return AVERROR(EINVAL);

    }



    if (avctx->sample_rate != 8000 && avctx->sample_rate != 16000 &&

        avctx->sample_rate != 11025 &&

        avctx->sample_rate != 22050 && avctx->sample_rate != 44100 &&

        avctx->strict_std_compliance >= FF_COMPLIANCE_NORMAL) {

        av_log(avctx, AV_LOG_ERROR, "Nellymoser works only with 8000, 16000, 11025, 22050 and 44100 sample rate\n");

        return AVERROR(EINVAL);

    }



    avctx->frame_size = NELLY_SAMPLES;

    avctx->delay      = NELLY_BUF_LEN;

    ff_af_queue_init(avctx, &s->afq);

    s->avctx = avctx;

    if ((ret = ff_mdct_init(&s->mdct_ctx, 8, 0, 32768.0)) < 0)

        goto error;

    ff_dsputil_init(&s->dsp, avctx);



    /* Generate overlap window */

    ff_sine_window_init(ff_sine_128, 128);

    for (i = 0; i < POW_TABLE_SIZE; i++)

        pow_table[i] = -pow(2, -i / 2048.0 - 3.0 + POW_TABLE_OFFSET);



    if (s->avctx->trellis) {

        s->opt  = av_malloc(NELLY_BANDS * OPT_SIZE * sizeof(float  ));

        s->path = av_malloc(NELLY_BANDS * OPT_SIZE * sizeof(uint8_t));

        if (!s->opt || !s->path) {

            ret = AVERROR(ENOMEM);

            goto error;

        }

    }



#if FF_API_OLD_ENCODE_AUDIO

    avctx->coded_frame = avcodec_alloc_frame();

    if (!avctx->coded_frame) {

        ret = AVERROR(ENOMEM);

        goto error;

    }

#endif



    return 0;

error:

    encode_end(avctx);

    return ret;

}
