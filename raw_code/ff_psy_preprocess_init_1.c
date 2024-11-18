av_cold struct FFPsyPreprocessContext* ff_psy_preprocess_init(AVCodecContext *avctx)

{

    FFPsyPreprocessContext *ctx;

    int i;

    float cutoff_coeff = 0;

    ctx        = av_mallocz(sizeof(FFPsyPreprocessContext));

    ctx->avctx = avctx;



    if (avctx->cutoff > 0)

        cutoff_coeff = 2.0 * avctx->cutoff / avctx->sample_rate;



    if (cutoff_coeff)

    ctx->fcoeffs = ff_iir_filter_init_coeffs(FF_FILTER_TYPE_BUTTERWORTH, FF_FILTER_MODE_LOWPASS,

                                             FILT_ORDER, cutoff_coeff, 0.0, 0.0);

    if (ctx->fcoeffs) {

        ctx->fstate = av_mallocz(sizeof(ctx->fstate[0]) * avctx->channels);

        for (i = 0; i < avctx->channels; i++)

            ctx->fstate[i] = ff_iir_filter_init_state(FILT_ORDER);

    }

    return ctx;

}
