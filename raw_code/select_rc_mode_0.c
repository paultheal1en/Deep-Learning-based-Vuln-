static int select_rc_mode(AVCodecContext *avctx, QSVEncContext *q)

{

    const char *rc_desc;

    mfxU16      rc_mode;



    int want_la     = q->la_depth >= 0;

    int want_qscale = !!(avctx->flags & AV_CODEC_FLAG_QSCALE);

    int want_vcm    = q->vcm;



    if (want_la && !QSV_HAVE_LA) {

        av_log(avctx, AV_LOG_ERROR,

               "Lookahead ratecontrol mode requested, but is not supported by this SDK version\n");

        return AVERROR(ENOSYS);

    }

    if (want_vcm && !QSV_HAVE_VCM) {

        av_log(avctx, AV_LOG_ERROR,

               "VCM ratecontrol mode requested, but is not supported by this SDK version\n");

        return AVERROR(ENOSYS);

    }



    if (want_la + want_qscale + want_vcm > 1) {

        av_log(avctx, AV_LOG_ERROR,

               "More than one of: { constant qscale, lookahead, VCM } requested, "

               "only one of them can be used at a time.\n");

        return AVERROR(EINVAL);

    }



    if (want_qscale) {

        rc_mode = MFX_RATECONTROL_CQP;

        rc_desc = "constant quantization parameter (CQP)";

    }

#if QSV_HAVE_VCM

    else if (want_vcm) {

        rc_mode = MFX_RATECONTROL_VCM;

        rc_desc = "video conferencing mode (VCM)";

    }

#endif

#if QSV_HAVE_LA

    else if (want_la) {

        rc_mode = MFX_RATECONTROL_LA;

        rc_desc = "VBR with lookahead (LA)";



#if QSV_HAVE_ICQ

        if (avctx->global_quality > 0) {

            rc_mode = MFX_RATECONTROL_LA_ICQ;

            rc_desc = "intelligent constant quality with lookahead (LA_ICQ)";

        }

#endif

    }

#endif

#if QSV_HAVE_ICQ

    else if (avctx->global_quality > 0) {

        rc_mode = MFX_RATECONTROL_ICQ;

        rc_desc = "intelligent constant quality (ICQ)";

    }

#endif

    else if (avctx->rc_max_rate == avctx->bit_rate) {

        rc_mode = MFX_RATECONTROL_CBR;

        rc_desc = "constant bitrate (CBR)";

    } else if (!avctx->rc_max_rate) {

        rc_mode = MFX_RATECONTROL_AVBR;

        rc_desc = "average variable bitrate (AVBR)";

    } else {

        rc_mode = MFX_RATECONTROL_VBR;

        rc_desc = "variable bitrate (VBR)";

    }



    q->param.mfx.RateControlMethod = rc_mode;

    av_log(avctx, AV_LOG_VERBOSE, "Using the %s ratecontrol method\n", rc_desc);



    return 0;

}
