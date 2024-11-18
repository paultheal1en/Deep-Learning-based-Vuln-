static int set_pix_fmt(AVCodecContext *avctx, vpx_codec_caps_t codec_caps,

                       struct vpx_codec_enc_cfg *enccfg, vpx_codec_flags_t *flags,

                       vpx_img_fmt_t *img_fmt)

{

    VPxContext av_unused *ctx = avctx->priv_data;

#ifdef VPX_IMG_FMT_HIGHBITDEPTH

    enccfg->g_bit_depth = enccfg->g_input_bit_depth = 8;

#endif

    switch (avctx->pix_fmt) {

    case AV_PIX_FMT_YUV420P:

    case AV_PIX_FMT_YUVA420P:

        enccfg->g_profile = 0;

        *img_fmt = VPX_IMG_FMT_I420;

        return 0;

    case AV_PIX_FMT_YUV422P:

        enccfg->g_profile = 1;

        *img_fmt = VPX_IMG_FMT_I422;

        return 0;

#if VPX_IMAGE_ABI_VERSION >= 3

    case AV_PIX_FMT_YUV440P:

        enccfg->g_profile = 1;

        *img_fmt = VPX_IMG_FMT_I440;

        return 0;

    case AV_PIX_FMT_GBRP:

        ctx->vpx_cs = VPX_CS_SRGB;

#endif

    case AV_PIX_FMT_YUV444P:

        enccfg->g_profile = 1;

        *img_fmt = VPX_IMG_FMT_I444;

        return 0;

#ifdef VPX_IMG_FMT_HIGHBITDEPTH

    case AV_PIX_FMT_YUV420P10:

    case AV_PIX_FMT_YUV420P12:

        if (codec_caps & VPX_CODEC_CAP_HIGHBITDEPTH) {

            enccfg->g_bit_depth = enccfg->g_input_bit_depth =

                avctx->pix_fmt == AV_PIX_FMT_YUV420P10 ? 10 : 12;

            enccfg->g_profile = 2;

            *img_fmt = VPX_IMG_FMT_I42016;

            *flags |= VPX_CODEC_USE_HIGHBITDEPTH;

            return 0;

        }

        break;

    case AV_PIX_FMT_YUV422P10:

    case AV_PIX_FMT_YUV422P12:

        if (codec_caps & VPX_CODEC_CAP_HIGHBITDEPTH) {

            enccfg->g_bit_depth = enccfg->g_input_bit_depth =

                avctx->pix_fmt == AV_PIX_FMT_YUV422P10 ? 10 : 12;

            enccfg->g_profile = 3;

            *img_fmt = VPX_IMG_FMT_I42216;

            *flags |= VPX_CODEC_USE_HIGHBITDEPTH;

            return 0;

        }

        break;

#if VPX_IMAGE_ABI_VERSION >= 3

    case AV_PIX_FMT_YUV440P10:

    case AV_PIX_FMT_YUV440P12:

        if (codec_caps & VPX_CODEC_CAP_HIGHBITDEPTH) {

            enccfg->g_bit_depth = enccfg->g_input_bit_depth =

                avctx->pix_fmt == AV_PIX_FMT_YUV440P10 ? 10 : 12;

            enccfg->g_profile = 3;

            *img_fmt = VPX_IMG_FMT_I44016;

            *flags |= VPX_CODEC_USE_HIGHBITDEPTH;

            return 0;

        }

        break;

    case AV_PIX_FMT_GBRP10:

    case AV_PIX_FMT_GBRP12:

        ctx->vpx_cs = VPX_CS_SRGB;

#endif

    case AV_PIX_FMT_YUV444P10:

    case AV_PIX_FMT_YUV444P12:

        if (codec_caps & VPX_CODEC_CAP_HIGHBITDEPTH) {

            enccfg->g_bit_depth = enccfg->g_input_bit_depth =

                avctx->pix_fmt == AV_PIX_FMT_YUV444P10 ||

                avctx->pix_fmt == AV_PIX_FMT_GBRP10 ? 10 : 12;

            enccfg->g_profile = 3;

            *img_fmt = VPX_IMG_FMT_I44416;

            *flags |= VPX_CODEC_USE_HIGHBITDEPTH;

            return 0;

        }

        break;

#endif

    default:

        break;

    }

    av_log(avctx, AV_LOG_ERROR, "Unsupported pixel format.\n");

    return AVERROR_INVALIDDATA;

}
