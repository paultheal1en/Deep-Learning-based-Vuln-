static int get_transform_coeffs(AC3DecodeContext * ctx)

{

    int i;

    ac3_audio_block *ab = &ctx->audio_block;

    float *samples = ctx->samples;

    int got_cplchan = 0;

    int dithflag = 0;



    samples += (ctx->bsi.flags & AC3_BSI_LFEON) ? 256 : 0;

    for (i = 0; i < ctx->bsi.nfchans; i++) {

        if ((ab->flags & AC3_AB_CPLINU) && (ab->chincpl & (1 << i)))

            dithflag = 0; /* don't generate dither until channels are decoupled */

        else

            dithflag = ab->dithflag & (1 << i);

        /* transform coefficients for individual channel */

        if (_get_transform_coeffs(ab->dexps[i], ab->bap[i], ab->chcoeffs[i], samples + (i * 256),

                    0, ab->endmant[i], dithflag, &ctx->gb, &ctx->state))

            return -1;

        /* tranform coefficients for coupling channels */

        if ((ab->flags & AC3_AB_CPLINU) && (ab->chincpl & (1 << i)) && !got_cplchan) {

            if (_get_transform_coeffs(ab->dcplexps, ab->cplbap, 1.0f, ab->cplcoeffs,

                        ab->cplstrtmant, ab->cplendmant, 0, &ctx->gb, &ctx->state))

                return -1;

            got_cplchan = 1;

        }

    }

    if (ctx->bsi.flags & AC3_BSI_LFEON)

        if (_get_transform_coeffs(ab->lfeexps, ab->lfebap, 1.0f, samples - 256, 0, 7, 0, &ctx->gb, &ctx->state))

                return -1;



    /* uncouple the channels from the coupling channel */

    if (ab->flags & AC3_AB_CPLINU)

        if (uncouple_channels(ctx))

            return -1;



    return 0;

}
