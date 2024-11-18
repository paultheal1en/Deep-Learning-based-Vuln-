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

    } else if (ctx->rc > 0) {

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

}
