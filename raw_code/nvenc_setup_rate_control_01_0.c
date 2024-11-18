static av_cold void nvenc_setup_rate_control(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;



    if (avctx->bit_rate > 0) {

        ctx->encode_config.rcParams.averageBitRate = avctx->bit_rate;

    } else if (ctx->encode_config.rcParams.averageBitRate > 0) {

        ctx->encode_config.rcParams.maxBitRate = ctx->encode_config.rcParams.averageBitRate;

    }



    if (avctx->rc_max_rate > 0)

        ctx->encode_config.rcParams.maxBitRate = avctx->rc_max_rate;



    if (ctx->rc < 0) {

        if (ctx->flags & NVENC_ONE_PASS)

            ctx->twopass = 0;

        if (ctx->flags & NVENC_TWO_PASSES)

            ctx->twopass = 1;



        if (ctx->twopass < 0)

            ctx->twopass = (ctx->flags & NVENC_LOWLATENCY) != 0;



        if (ctx->cbr) {

            if (ctx->twopass) {

                ctx->rc = NV_ENC_PARAMS_RC_2_PASS_QUALITY;

            } else {

                ctx->rc = NV_ENC_PARAMS_RC_CBR;

            }

        } else if (avctx->global_quality > 0) {

            ctx->rc = NV_ENC_PARAMS_RC_CONSTQP;

        } else if (ctx->twopass) {

            ctx->rc = NV_ENC_PARAMS_RC_2_PASS_VBR;

        } else if (avctx->qmin >= 0 && avctx->qmax >= 0) {

            ctx->rc = NV_ENC_PARAMS_RC_VBR_MINQP;

        }

    }



    if (ctx->flags & NVENC_LOSSLESS) {

        set_lossless(avctx);

    } else if (ctx->rc >= 0) {

        nvenc_override_rate_control(avctx);

    } else {

        ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

        set_vbr(avctx);

    }



    if (avctx->rc_buffer_size > 0) {

        ctx->encode_config.rcParams.vbvBufferSize = avctx->rc_buffer_size;

    } else if (ctx->encode_config.rcParams.averageBitRate > 0) {

        ctx->encode_config.rcParams.vbvBufferSize = 2 * ctx->encode_config.rcParams.averageBitRate;

    }



    if (ctx->aq) {

        ctx->encode_config.rcParams.enableAQ   = 1;

        ctx->encode_config.rcParams.aqStrength = ctx->aq_strength;

        av_log(avctx, AV_LOG_VERBOSE, "AQ enabled.\n");

    }



    if (ctx->temporal_aq) {

        ctx->encode_config.rcParams.enableTemporalAQ = 1;

        av_log(avctx, AV_LOG_VERBOSE, "Temporal AQ enabled.\n");

    }



    if (ctx->rc_lookahead) {

        int lkd_bound = FFMIN(ctx->nb_surfaces, ctx->async_depth) -

                        ctx->encode_config.frameIntervalP - 4;



        if (lkd_bound < 0) {

            av_log(avctx, AV_LOG_WARNING,

                   "Lookahead not enabled. Increase buffer delay (-delay).\n");

        } else {

            ctx->encode_config.rcParams.enableLookahead = 1;

            ctx->encode_config.rcParams.lookaheadDepth  = av_clip(ctx->rc_lookahead, 0, lkd_bound);

            ctx->encode_config.rcParams.disableIadapt   = ctx->no_scenecut;

            ctx->encode_config.rcParams.disableBadapt   = !ctx->b_adapt;

            av_log(avctx, AV_LOG_VERBOSE,

                   "Lookahead enabled: depth %d, scenecut %s, B-adapt %s.\n",

                   ctx->encode_config.rcParams.lookaheadDepth,

                   ctx->encode_config.rcParams.disableIadapt ? "disabled" : "enabled",

                   ctx->encode_config.rcParams.disableBadapt ? "disabled" : "enabled");

        }

    }



    if (ctx->strict_gop) {

        ctx->encode_config.rcParams.strictGOPTarget = 1;

        av_log(avctx, AV_LOG_VERBOSE, "Strict GOP target enabled.\n");

    }



    if (ctx->nonref_p)

        ctx->encode_config.rcParams.enableNonRefP = 1;



    if (ctx->zerolatency)

        ctx->encode_config.rcParams.zeroReorderDelay = 1;



    if (ctx->quality)

        ctx->encode_config.rcParams.targetQuality = ctx->quality;

}
