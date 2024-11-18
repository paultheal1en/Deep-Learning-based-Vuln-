static void dump_video_param(AVCodecContext *avctx, QSVEncContext *q,

                             mfxExtBuffer **coding_opts)

{

    mfxInfoMFX *info = &q->param.mfx;



    mfxExtCodingOption   *co = (mfxExtCodingOption*)coding_opts[0];

#if QSV_HAVE_CO2

    mfxExtCodingOption2 *co2 = (mfxExtCodingOption2*)coding_opts[1];

#endif

#if QSV_HAVE_CO3

    mfxExtCodingOption3 *co3 = (mfxExtCodingOption3*)coding_opts[2];

#endif



    av_log(avctx, AV_LOG_VERBOSE, "profile: %s; level: %"PRIu16"\n",

           print_profile(info->CodecProfile), info->CodecLevel);



    av_log(avctx, AV_LOG_VERBOSE, "GopPicSize: %"PRIu16"; GopRefDist: %"PRIu16"; GopOptFlag: ",

           info->GopPicSize, info->GopRefDist);

    if (info->GopOptFlag & MFX_GOP_CLOSED)

        av_log(avctx, AV_LOG_VERBOSE, "closed ");

    if (info->GopOptFlag & MFX_GOP_STRICT)

        av_log(avctx, AV_LOG_VERBOSE, "strict ");

    av_log(avctx, AV_LOG_VERBOSE, "; IdrInterval: %"PRIu16"\n", info->IdrInterval);



    av_log(avctx, AV_LOG_VERBOSE, "TargetUsage: %"PRIu16"; RateControlMethod: %s\n",

           info->TargetUsage, print_ratecontrol(info->RateControlMethod));



    if (info->RateControlMethod == MFX_RATECONTROL_CBR ||

        info->RateControlMethod == MFX_RATECONTROL_VBR

#if QSV_HAVE_VCM

        || info->RateControlMethod == MFX_RATECONTROL_VCM

#endif

        ) {

        av_log(avctx, AV_LOG_VERBOSE,

               "InitialDelayInKB: %"PRIu16"; TargetKbps: %"PRIu16"; MaxKbps: %"PRIu16"\n",

               info->InitialDelayInKB, info->TargetKbps, info->MaxKbps);

    } else if (info->RateControlMethod == MFX_RATECONTROL_CQP) {

        av_log(avctx, AV_LOG_VERBOSE, "QPI: %"PRIu16"; QPP: %"PRIu16"; QPB: %"PRIu16"\n",

               info->QPI, info->QPP, info->QPB);

    } else if (info->RateControlMethod == MFX_RATECONTROL_AVBR) {

        av_log(avctx, AV_LOG_VERBOSE,

               "TargetKbps: %"PRIu16"; Accuracy: %"PRIu16"; Convergence: %"PRIu16"\n",

               info->TargetKbps, info->Accuracy, info->Convergence);

    }

#if QSV_HAVE_LA

    else if (info->RateControlMethod == MFX_RATECONTROL_LA

#if QSV_HAVE_LA_HRD

             || info->RateControlMethod == MFX_RATECONTROL_LA_HRD

#endif

             ) {

        av_log(avctx, AV_LOG_VERBOSE,

               "TargetKbps: %"PRIu16"; LookAheadDepth: %"PRIu16"\n",

               info->TargetKbps, co2->LookAheadDepth);

    }

#endif

#if QSV_HAVE_ICQ

    else if (info->RateControlMethod == MFX_RATECONTROL_ICQ) {

        av_log(avctx, AV_LOG_VERBOSE, "ICQQuality: %"PRIu16"\n", info->ICQQuality);

    } else if (info->RateControlMethod == MFX_RATECONTROL_LA_ICQ) {

        av_log(avctx, AV_LOG_VERBOSE, "ICQQuality: %"PRIu16"; LookAheadDepth: %"PRIu16"\n",

               info->ICQQuality, co2->LookAheadDepth);

    }

#endif

#if QSV_HAVE_QVBR

    else if (info->RateControlMethod == MFX_RATECONTROL_QVBR) {

        av_log(avctx, AV_LOG_VERBOSE, "QVBRQuality: %"PRIu16"\n",

               co3->QVBRQuality);

    }

#endif



    av_log(avctx, AV_LOG_VERBOSE, "NumSlice: %"PRIu16"; NumRefFrame: %"PRIu16"\n",

           info->NumSlice, info->NumRefFrame);

    av_log(avctx, AV_LOG_VERBOSE, "RateDistortionOpt: %s\n",

           print_threestate(co->RateDistortionOpt));



#if QSV_HAVE_CO2

    av_log(avctx, AV_LOG_VERBOSE,

           "RecoveryPointSEI: %s IntRefType: %"PRIu16"; IntRefCycleSize: %"PRIu16"; IntRefQPDelta: %"PRId16"\n",

           print_threestate(co->RecoveryPointSEI), co2->IntRefType, co2->IntRefCycleSize, co2->IntRefQPDelta);



    av_log(avctx, AV_LOG_VERBOSE, "MaxFrameSize: %"PRIu16"; ", co2->MaxFrameSize);

#if QSV_VERSION_ATLEAST(1, 9)

    av_log(avctx, AV_LOG_VERBOSE, "MaxSliceSize: %"PRIu16"; ", co2->MaxSliceSize);

#endif

    av_log(avctx, AV_LOG_VERBOSE, "\n");



    av_log(avctx, AV_LOG_VERBOSE,

           "BitrateLimit: %s; MBBRC: %s; ExtBRC: %s\n",

           print_threestate(co2->BitrateLimit), print_threestate(co2->MBBRC),

           print_threestate(co2->ExtBRC));



#if QSV_HAVE_TRELLIS

    av_log(avctx, AV_LOG_VERBOSE, "Trellis: ");

    if (co2->Trellis & MFX_TRELLIS_OFF) {

        av_log(avctx, AV_LOG_VERBOSE, "off");

    } else if (!co2->Trellis) {

        av_log(avctx, AV_LOG_VERBOSE, "auto");

    } else {

        if (co2->Trellis & MFX_TRELLIS_I) av_log(avctx, AV_LOG_VERBOSE, "I");

        if (co2->Trellis & MFX_TRELLIS_P) av_log(avctx, AV_LOG_VERBOSE, "P");

        if (co2->Trellis & MFX_TRELLIS_B) av_log(avctx, AV_LOG_VERBOSE, "B");

    }

    av_log(avctx, AV_LOG_VERBOSE, "\n");

#endif



#if QSV_VERSION_ATLEAST(1, 8)

    av_log(avctx, AV_LOG_VERBOSE,

           "RepeatPPS: %s; NumMbPerSlice: %"PRIu16"; LookAheadDS: ",

           print_threestate(co2->RepeatPPS), co2->NumMbPerSlice);

    switch (co2->LookAheadDS) {

    case MFX_LOOKAHEAD_DS_OFF: av_log(avctx, AV_LOG_VERBOSE, "off");     break;

    case MFX_LOOKAHEAD_DS_2x:  av_log(avctx, AV_LOG_VERBOSE, "2x");      break;

    case MFX_LOOKAHEAD_DS_4x:  av_log(avctx, AV_LOG_VERBOSE, "4x");      break;

    default:                   av_log(avctx, AV_LOG_VERBOSE, "unknown"); break;

    }

    av_log(avctx, AV_LOG_VERBOSE, "\n");



    av_log(avctx, AV_LOG_VERBOSE, "AdaptiveI: %s; AdaptiveB: %s; BRefType: ",

           print_threestate(co2->AdaptiveI), print_threestate(co2->AdaptiveB));

    switch (co2->BRefType) {

    case MFX_B_REF_OFF:     av_log(avctx, AV_LOG_VERBOSE, "off");       break;

    case MFX_B_REF_PYRAMID: av_log(avctx, AV_LOG_VERBOSE, "pyramid");   break;

    default:                av_log(avctx, AV_LOG_VERBOSE, "auto");      break;

    }

    av_log(avctx, AV_LOG_VERBOSE, "\n");

#endif



#if QSV_VERSION_ATLEAST(1, 9)

    av_log(avctx, AV_LOG_VERBOSE,

           "MinQPI: %"PRIu8"; MaxQPI: %"PRIu8"; MinQPP: %"PRIu8"; MaxQPP: %"PRIu8"; MinQPB: %"PRIu8"; MaxQPB: %"PRIu8"\n",

           co2->MinQPI, co2->MaxQPI, co2->MinQPP, co2->MaxQPP, co2->MinQPB, co2->MaxQPB);

#endif

#endif



    if (avctx->codec_id == AV_CODEC_ID_H264) {

        av_log(avctx, AV_LOG_VERBOSE, "Entropy coding: %s; MaxDecFrameBuffering: %"PRIu16"\n",

               co->CAVLC == MFX_CODINGOPTION_ON ? "CAVLC" : "CABAC", co->MaxDecFrameBuffering);

        av_log(avctx, AV_LOG_VERBOSE,

               "NalHrdConformance: %s; SingleSeiNalUnit: %s; VuiVclHrdParameters: %s VuiNalHrdParameters: %s\n",

               print_threestate(co->NalHrdConformance), print_threestate(co->SingleSeiNalUnit),

               print_threestate(co->VuiVclHrdParameters), print_threestate(co->VuiNalHrdParameters));

    }

}
