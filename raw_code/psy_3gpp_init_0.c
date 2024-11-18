static av_cold int psy_3gpp_init(FFPsyContext *ctx) {

    AacPsyContext *pctx;

    float bark;

    int i, j, g, start;

    float prev, minscale, minath;



    ctx->model_priv_data = av_mallocz(sizeof(AacPsyContext));

    pctx = (AacPsyContext*) ctx->model_priv_data;



    minath = ath(3410, ATH_ADD);

    for (j = 0; j < 2; j++) {

        AacPsyCoeffs *coeffs = &pctx->psy_coef[j];

        float line_to_frequency = ctx->avctx->sample_rate / (j ? 256.f : 2048.0f);

        i = 0;

        prev = 0.0;

        for (g = 0; g < ctx->num_bands[j]; g++) {

            i += ctx->bands[j][g];

            bark = calc_bark((i-1) * line_to_frequency);

            coeffs->barks[g] = (bark + prev) / 2.0;

            prev = bark;

        }

        for (g = 0; g < ctx->num_bands[j] - 1; g++) {

            coeffs->spread_low[g] = pow(10.0, -(coeffs->barks[g+1] - coeffs->barks[g]) * PSY_3GPP_SPREAD_LOW);

            coeffs->spread_hi [g] = pow(10.0, -(coeffs->barks[g+1] - coeffs->barks[g]) * PSY_3GPP_SPREAD_HI);

        }

        start = 0;

        for (g = 0; g < ctx->num_bands[j]; g++) {

            minscale = ath(start * line_to_frequency, ATH_ADD);

            for (i = 1; i < ctx->bands[j][g]; i++)

                minscale = FFMIN(minscale, ath((start + i) * line_to_frequency, ATH_ADD));

            coeffs->ath[g] = minscale - minath;

            start += ctx->bands[j][g];

        }

    }



    pctx->ch = av_mallocz(sizeof(AacPsyChannel) * ctx->avctx->channels);



    lame_window_init(pctx, ctx->avctx);



    return 0;

}
