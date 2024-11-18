static av_cold int psy_3gpp_init(FFPsyContext *ctx) {

    AacPsyContext *pctx;

    float bark;

    int i, j, g, start;

    float prev, minscale, minath, minsnr, pe_min;

    int chan_bitrate = ctx->avctx->bit_rate / ((ctx->avctx->flags & CODEC_FLAG_QSCALE) ? 2.0f : ctx->avctx->channels);



    const int bandwidth    = ctx->avctx->cutoff ? ctx->avctx->cutoff : AAC_CUTOFF(ctx->avctx);

    const float num_bark   = calc_bark((float)bandwidth);



    ctx->model_priv_data = av_mallocz(sizeof(AacPsyContext));

    if (!ctx->model_priv_data)

        return AVERROR(ENOMEM);

    pctx = (AacPsyContext*) ctx->model_priv_data;

    pctx->global_quality = (ctx->avctx->global_quality ? ctx->avctx->global_quality : 120) * 0.01f;



    if (ctx->avctx->flags & CODEC_FLAG_QSCALE) {

        /* Use the target average bitrate to compute spread parameters */

        chan_bitrate = (int)(chan_bitrate / 120.0 * (ctx->avctx->global_quality ? ctx->avctx->global_quality : 120));

    }



    pctx->chan_bitrate = chan_bitrate;

    pctx->frame_bits   = FFMIN(2560, chan_bitrate * AAC_BLOCK_SIZE_LONG / ctx->avctx->sample_rate);

    pctx->pe.min       =  8.0f * AAC_BLOCK_SIZE_LONG * bandwidth / (ctx->avctx->sample_rate * 2.0f);

    pctx->pe.max       = 12.0f * AAC_BLOCK_SIZE_LONG * bandwidth / (ctx->avctx->sample_rate * 2.0f);

    ctx->bitres.size   = 6144 - pctx->frame_bits;

    ctx->bitres.size  -= ctx->bitres.size % 8;

    pctx->fill_level   = ctx->bitres.size;

    minath = ath(3410 - 0.733 * ATH_ADD, ATH_ADD);

    for (j = 0; j < 2; j++) {

        AacPsyCoeffs *coeffs = pctx->psy_coef[j];

        const uint8_t *band_sizes = ctx->bands[j];

        float line_to_frequency = ctx->avctx->sample_rate / (j ? 256.f : 2048.0f);

        float avg_chan_bits = chan_bitrate * (j ? 128.0f : 1024.0f) / ctx->avctx->sample_rate;

        /* reference encoder uses 2.4% here instead of 60% like the spec says */

        float bark_pe = 0.024f * PSY_3GPP_BITS_TO_PE(avg_chan_bits) / num_bark;

        float en_spread_low = j ? PSY_3GPP_EN_SPREAD_LOW_S : PSY_3GPP_EN_SPREAD_LOW_L;

        /* High energy spreading for long blocks <= 22kbps/channel and short blocks are the same. */

        float en_spread_hi  = (j || (chan_bitrate <= 22.0f)) ? PSY_3GPP_EN_SPREAD_HI_S : PSY_3GPP_EN_SPREAD_HI_L1;



        i = 0;

        prev = 0.0;

        for (g = 0; g < ctx->num_bands[j]; g++) {

            i += band_sizes[g];

            bark = calc_bark((i-1) * line_to_frequency);

            coeffs[g].barks = (bark + prev) / 2.0;

            prev = bark;

        }

        for (g = 0; g < ctx->num_bands[j] - 1; g++) {

            AacPsyCoeffs *coeff = &coeffs[g];

            float bark_width = coeffs[g+1].barks - coeffs->barks;

            coeff->spread_low[0] = pow(10.0, -bark_width * PSY_3GPP_THR_SPREAD_LOW);

            coeff->spread_hi [0] = pow(10.0, -bark_width * PSY_3GPP_THR_SPREAD_HI);

            coeff->spread_low[1] = pow(10.0, -bark_width * en_spread_low);

            coeff->spread_hi [1] = pow(10.0, -bark_width * en_spread_hi);

            pe_min = bark_pe * bark_width;

            minsnr = exp2(pe_min / band_sizes[g]) - 1.5f;

            coeff->min_snr = av_clipf(1.0f / minsnr, PSY_SNR_25DB, PSY_SNR_1DB);

        }

        start = 0;

        for (g = 0; g < ctx->num_bands[j]; g++) {

            minscale = ath(start * line_to_frequency, ATH_ADD);

            for (i = 1; i < band_sizes[g]; i++)

                minscale = FFMIN(minscale, ath((start + i) * line_to_frequency, ATH_ADD));

            coeffs[g].ath = minscale - minath;

            start += band_sizes[g];

        }

    }



    pctx->ch = av_mallocz_array(ctx->avctx->channels, sizeof(AacPsyChannel));

    if (!pctx->ch) {

        av_freep(&ctx->model_priv_data);

        return AVERROR(ENOMEM);

    }



    lame_window_init(pctx, ctx->avctx);



    return 0;

}
