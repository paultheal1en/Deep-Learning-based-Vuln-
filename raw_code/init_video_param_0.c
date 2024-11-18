static int init_video_param(AVCodecContext *avctx, QSVEncContext *q)

{

    float quant;

    int ret;



    ret = ff_qsv_codec_id_to_mfx(avctx->codec_id);

    if (ret < 0)

        return AVERROR_BUG;

    q->param.mfx.CodecId = ret;



    q->width_align = avctx->codec_id == AV_CODEC_ID_HEVC ? 32 : 16;



    if (avctx->level > 0)

        q->param.mfx.CodecLevel = avctx->level;



    q->param.mfx.CodecProfile       = q->profile;

    q->param.mfx.TargetUsage        = q->preset;

    q->param.mfx.GopPicSize         = FFMAX(0, avctx->gop_size);

    q->param.mfx.GopRefDist         = FFMAX(-1, avctx->max_b_frames) + 1;

    q->param.mfx.GopOptFlag         = avctx->flags & AV_CODEC_FLAG_CLOSED_GOP ?

                                      MFX_GOP_CLOSED : 0;

    q->param.mfx.IdrInterval        = q->idr_interval;

    q->param.mfx.NumSlice           = avctx->slices;

    q->param.mfx.NumRefFrame        = FFMAX(0, avctx->refs);

    q->param.mfx.EncodedOrder       = 0;

    q->param.mfx.BufferSizeInKB     = 0;



    q->param.mfx.FrameInfo.FourCC         = MFX_FOURCC_NV12;

    q->param.mfx.FrameInfo.Width          = FFALIGN(avctx->width, q->width_align);

    q->param.mfx.FrameInfo.Height         = FFALIGN(avctx->height, 32);

    q->param.mfx.FrameInfo.CropX          = 0;

    q->param.mfx.FrameInfo.CropY          = 0;

    q->param.mfx.FrameInfo.CropW          = avctx->width;

    q->param.mfx.FrameInfo.CropH          = avctx->height;

    q->param.mfx.FrameInfo.AspectRatioW   = avctx->sample_aspect_ratio.num;

    q->param.mfx.FrameInfo.AspectRatioH   = avctx->sample_aspect_ratio.den;

    q->param.mfx.FrameInfo.PicStruct      = MFX_PICSTRUCT_PROGRESSIVE;

    q->param.mfx.FrameInfo.ChromaFormat   = MFX_CHROMAFORMAT_YUV420;

    q->param.mfx.FrameInfo.BitDepthLuma   = 8;

    q->param.mfx.FrameInfo.BitDepthChroma = 8;



    if (avctx->framerate.den > 0 && avctx->framerate.num > 0) {

        q->param.mfx.FrameInfo.FrameRateExtN = avctx->framerate.num;

        q->param.mfx.FrameInfo.FrameRateExtD = avctx->framerate.den;

    } else {

        q->param.mfx.FrameInfo.FrameRateExtN  = avctx->time_base.den;

        q->param.mfx.FrameInfo.FrameRateExtD  = avctx->time_base.num;

    }



    ret = select_rc_mode(avctx, q);

    if (ret < 0)

        return ret;



    switch (q->param.mfx.RateControlMethod) {

    case MFX_RATECONTROL_CBR:

    case MFX_RATECONTROL_VBR:

#if QSV_HAVE_VCM

    case MFX_RATECONTROL_VCM:

#endif

        q->param.mfx.InitialDelayInKB = avctx->rc_initial_buffer_occupancy / 1000;

        q->param.mfx.TargetKbps       = avctx->bit_rate / 1000;

        q->param.mfx.MaxKbps          = avctx->rc_max_rate / 1000;

        break;

    case MFX_RATECONTROL_CQP:

        quant = avctx->global_quality / FF_QP2LAMBDA;



        q->param.mfx.QPI = av_clip(quant * fabs(avctx->i_quant_factor) + avctx->i_quant_offset, 0, 51);

        q->param.mfx.QPP = av_clip(quant, 0, 51);

        q->param.mfx.QPB = av_clip(quant * fabs(avctx->b_quant_factor) + avctx->b_quant_offset, 0, 51);



        break;

    case MFX_RATECONTROL_AVBR:

        q->param.mfx.TargetKbps  = avctx->bit_rate / 1000;

        q->param.mfx.Convergence = q->avbr_convergence;

        q->param.mfx.Accuracy    = q->avbr_accuracy;

        break;

#if QSV_HAVE_LA

    case MFX_RATECONTROL_LA:

        q->param.mfx.TargetKbps  = avctx->bit_rate / 1000;

        q->extco2.LookAheadDepth = q->la_depth;

        break;

#if QSV_HAVE_ICQ

    case MFX_RATECONTROL_LA_ICQ:

        q->extco2.LookAheadDepth = q->la_depth;

    case MFX_RATECONTROL_ICQ:

        q->param.mfx.ICQQuality  = avctx->global_quality;

        break;

#endif

#endif

    }



    // the HEVC encoder plugin currently fails if coding options

    // are provided

    if (avctx->codec_id != AV_CODEC_ID_HEVC) {

        q->extco.Header.BufferId      = MFX_EXTBUFF_CODING_OPTION;

        q->extco.Header.BufferSz      = sizeof(q->extco);

        q->extco.CAVLC                = avctx->coder_type == FF_CODER_TYPE_VLC ?

                                        MFX_CODINGOPTION_ON : MFX_CODINGOPTION_UNKNOWN;



        if (q->rdo >= 0)

            q->extco.RateDistortionOpt = q->rdo > 0 ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;



        if (avctx->codec_id == AV_CODEC_ID_H264) {

            if (avctx->strict_std_compliance != FF_COMPLIANCE_NORMAL)

                q->extco.NalHrdConformance = avctx->strict_std_compliance > FF_COMPLIANCE_NORMAL ?

                                             MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;



            if (q->single_sei_nal_unit >= 0)

                q->extco.SingleSeiNalUnit = q->single_sei_nal_unit ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

            if (q->recovery_point_sei >= 0)

                q->extco.RecoveryPointSEI = q->recovery_point_sei ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

            q->extco.MaxDecFrameBuffering = q->max_dec_frame_buffering;

        }



        q->extparam_internal[q->nb_extparam_internal++] = (mfxExtBuffer *)&q->extco;



#if QSV_HAVE_CO2

        if (avctx->codec_id == AV_CODEC_ID_H264) {

            q->extco2.Header.BufferId     = MFX_EXTBUFF_CODING_OPTION2;

            q->extco2.Header.BufferSz     = sizeof(q->extco2);



            if (q->int_ref_type >= 0)

                q->extco2.IntRefType = q->int_ref_type;

            if (q->int_ref_cycle_size >= 0)

                q->extco2.IntRefCycleSize = q->int_ref_cycle_size;

            if (q->int_ref_qp_delta != INT16_MIN)

                q->extco2.IntRefQPDelta = q->int_ref_qp_delta;



            if (q->bitrate_limit >= 0)

                q->extco2.BitrateLimit = q->bitrate_limit ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

            if (q->mbbrc >= 0)

                q->extco2.MBBRC = q->mbbrc ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

            if (q->extbrc >= 0)

                q->extco2.ExtBRC = q->extbrc ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;



            if (q->max_frame_size >= 0)

                q->extco2.MaxFrameSize = q->max_frame_size;

#if QSV_HAVE_MAX_SLICE_SIZE

            if (q->max_slice_size >= 0)

                q->extco2.MaxSliceSize = q->max_slice_size;

#endif



#if QSV_HAVE_TRELLIS

            q->extco2.Trellis = q->trellis;

#endif



#if QSV_HAVE_BREF_TYPE

            if (avctx->b_frame_strategy >= 0)

                q->extco2.BRefType = avctx->b_frame_strategy ? MFX_B_REF_PYRAMID : MFX_B_REF_OFF;

            if (q->adaptive_i >= 0)

                q->extco2.AdaptiveI = q->adaptive_i ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

            if (q->adaptive_b >= 0)

                q->extco2.AdaptiveB = q->adaptive_b ? MFX_CODINGOPTION_ON : MFX_CODINGOPTION_OFF;

#endif



            q->extparam_internal[q->nb_extparam_internal++] = (mfxExtBuffer *)&q->extco2;

        }

#endif

    }



    if (!rc_supported(q)) {

        av_log(avctx, AV_LOG_ERROR,

               "Selected ratecontrol mode is not supported by the QSV "

               "runtime. Choose a different mode.\n");

        return AVERROR(ENOSYS);

    }



    return 0;

}
