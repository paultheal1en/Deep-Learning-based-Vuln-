static av_cold int atrac1_decode_init(AVCodecContext *avctx)

{

    AT1Ctx *q = avctx->priv_data;



    avctx->sample_fmt = AV_SAMPLE_FMT_FLT;



    if (avctx->channels < 1 || avctx->channels > AT1_MAX_CHANNELS) {

        av_log(avctx, AV_LOG_ERROR, "Unsupported number of channels: %d\n",

               avctx->channels);

        return AVERROR(EINVAL);

    }

    q->channels = avctx->channels;



    if (avctx->channels == 2) {

        q->out_samples[0] = av_malloc(2 * AT1_SU_SAMPLES * sizeof(*q->out_samples[0]));

        q->out_samples[1] = q->out_samples[0] + AT1_SU_SAMPLES;

        if (!q->out_samples[0]) {

            av_freep(&q->out_samples[0]);

            return AVERROR(ENOMEM);

        }

    }



    /* Init the mdct transforms */

    ff_mdct_init(&q->mdct_ctx[0], 6, 1, -1.0/ (1 << 15));

    ff_mdct_init(&q->mdct_ctx[1], 8, 1, -1.0/ (1 << 15));

    ff_mdct_init(&q->mdct_ctx[2], 9, 1, -1.0/ (1 << 15));



    ff_init_ff_sine_windows(5);



    atrac_generate_tables();



    dsputil_init(&q->dsp, avctx);

    ff_fmt_convert_init(&q->fmt_conv, avctx);



    q->bands[0] = q->low;

    q->bands[1] = q->mid;

    q->bands[2] = q->high;



    /* Prepare the mdct overlap buffers */

    q->SUs[0].spectrum[0] = q->SUs[0].spec1;

    q->SUs[0].spectrum[1] = q->SUs[0].spec2;

    q->SUs[1].spectrum[0] = q->SUs[1].spec1;

    q->SUs[1].spectrum[1] = q->SUs[1].spec2;



    return 0;

}
