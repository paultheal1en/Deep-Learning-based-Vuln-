static av_cold int nvenc_encode_init(AVCodecContext *avctx)

{

    NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS encode_session_params = { 0 };

    NV_ENC_PRESET_CONFIG preset_config = { 0 };

    CUcontext cu_context_curr;

    CUresult cu_res;

    GUID encoder_preset = NV_ENC_PRESET_HQ_GUID;

    GUID codec;

    NVENCSTATUS nv_status = NV_ENC_SUCCESS;

    AVCPBProperties *cpb_props;

    int surfaceCount = 0;

    int i, num_mbs;

    int isLL = 0;

    int lossless = 0;

    int res = 0;

    int dw, dh;

    int qp_inter_p;



    NvencContext *ctx = avctx->priv_data;

    NvencDynLoadFunctions *dl_fn = &ctx->nvenc_dload_funcs;

    NV_ENCODE_API_FUNCTION_LIST *p_nvenc = &dl_fn->nvenc_funcs;



    if (!nvenc_dyload_nvenc(avctx))

        return AVERROR_EXTERNAL;



    ctx->last_dts = AV_NOPTS_VALUE;



    ctx->encode_config.version = NV_ENC_CONFIG_VER;

    ctx->init_encode_params.version = NV_ENC_INITIALIZE_PARAMS_VER;

    preset_config.version = NV_ENC_PRESET_CONFIG_VER;

    preset_config.presetCfg.version = NV_ENC_CONFIG_VER;

    encode_session_params.version = NV_ENC_OPEN_ENCODE_SESSION_EX_PARAMS_VER;

    encode_session_params.apiVersion = NVENCAPI_VERSION;



    if (ctx->gpu >= dl_fn->nvenc_device_count) {

        av_log(avctx, AV_LOG_FATAL, "Requested GPU %d, but only %d GPUs are available!\n", ctx->gpu, dl_fn->nvenc_device_count);

        res = AVERROR(EINVAL);

        goto error;

    }



    ctx->cu_context = NULL;

    cu_res = dl_fn->cu_ctx_create(&ctx->cu_context, 4, dl_fn->nvenc_devices[ctx->gpu]); // CU_CTX_SCHED_BLOCKING_SYNC=4, avoid CPU spins



    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "Failed creating CUDA context for NVENC: 0x%x\n", (int)cu_res);

        res = AVERROR_EXTERNAL;

        goto error;

    }



    cu_res = dl_fn->cu_ctx_pop_current(&cu_context_curr);



    if (cu_res != CUDA_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "Failed popping CUDA context: 0x%x\n", (int)cu_res);

        res = AVERROR_EXTERNAL;

        goto error;

    }



    encode_session_params.device = ctx->cu_context;

    encode_session_params.deviceType = NV_ENC_DEVICE_TYPE_CUDA;



    nv_status = p_nvenc->nvEncOpenEncodeSessionEx(&encode_session_params, &ctx->nvencoder);

    if (nv_status != NV_ENC_SUCCESS) {

        ctx->nvencoder = NULL;

        av_log(avctx, AV_LOG_FATAL, "OpenEncodeSessionEx failed: 0x%x\n", (int)nv_status);

        res = AVERROR_EXTERNAL;

        goto error;

    }



    if (ctx->preset) {

        if (!strcmp(ctx->preset, "slow")) {

            encoder_preset = NV_ENC_PRESET_HQ_GUID;

            ctx->twopass = 1;

        } else if (!strcmp(ctx->preset, "medium")) {

            encoder_preset = NV_ENC_PRESET_HQ_GUID;

            ctx->twopass = 0;

        } else if (!strcmp(ctx->preset, "fast")) {

            encoder_preset = NV_ENC_PRESET_HP_GUID;

            ctx->twopass = 0;

        } else if (!strcmp(ctx->preset, "hq")) {

            encoder_preset = NV_ENC_PRESET_HQ_GUID;

        } else if (!strcmp(ctx->preset, "hp")) {

            encoder_preset = NV_ENC_PRESET_HP_GUID;

        } else if (!strcmp(ctx->preset, "bd")) {

            encoder_preset = NV_ENC_PRESET_BD_GUID;

        } else if (!strcmp(ctx->preset, "ll")) {

            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_DEFAULT_GUID;

            isLL = 1;

        } else if (!strcmp(ctx->preset, "llhp")) {

            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_HP_GUID;

            isLL = 1;

        } else if (!strcmp(ctx->preset, "llhq")) {

            encoder_preset = NV_ENC_PRESET_LOW_LATENCY_HQ_GUID;

            isLL = 1;

        } else if (!strcmp(ctx->preset, "lossless")) {

            encoder_preset = NV_ENC_PRESET_LOSSLESS_DEFAULT_GUID;

            lossless = 1;

        } else if (!strcmp(ctx->preset, "losslesshp")) {

            encoder_preset = NV_ENC_PRESET_LOSSLESS_HP_GUID;

            lossless = 1;

        } else if (!strcmp(ctx->preset, "default")) {

            encoder_preset = NV_ENC_PRESET_DEFAULT_GUID;

        } else {

            av_log(avctx, AV_LOG_FATAL, "Preset \"%s\" is unknown! Supported presets: slow, medium, fast, hp, hq, bd, ll, llhp, llhq, lossless, losslesshp, default\n", ctx->preset);

            res = AVERROR(EINVAL);

            goto error;

        }

    }



    if (ctx->twopass < 0) {

        ctx->twopass = isLL;

    }



    switch (avctx->codec->id) {

    case AV_CODEC_ID_H264:

        codec = NV_ENC_CODEC_H264_GUID;

        break;

    case AV_CODEC_ID_H265:

        codec = NV_ENC_CODEC_HEVC_GUID;

        break;

    default:

        av_log(avctx, AV_LOG_ERROR, "Unknown codec name\n");

        res = AVERROR(EINVAL);

        goto error;

    }



    nv_status = p_nvenc->nvEncGetEncodePresetConfig(ctx->nvencoder, codec, encoder_preset, &preset_config);

    if (nv_status != NV_ENC_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "GetEncodePresetConfig failed: 0x%x\n", (int)nv_status);

        res = AVERROR_EXTERNAL;

        goto error;

    }



    ctx->init_encode_params.encodeGUID = codec;

    ctx->init_encode_params.encodeHeight = avctx->height;

    ctx->init_encode_params.encodeWidth = avctx->width;



    if (avctx->sample_aspect_ratio.num && avctx->sample_aspect_ratio.den &&

        (avctx->sample_aspect_ratio.num != 1 || avctx->sample_aspect_ratio.num != 1)) {

        av_reduce(&dw, &dh,

                  avctx->width * avctx->sample_aspect_ratio.num,

                  avctx->height * avctx->sample_aspect_ratio.den,

                  1024 * 1024);

        ctx->init_encode_params.darHeight = dh;

        ctx->init_encode_params.darWidth = dw;

    } else {

        ctx->init_encode_params.darHeight = avctx->height;

        ctx->init_encode_params.darWidth = avctx->width;

    }



    // De-compensate for hardware, dubiously, trying to compensate for

    // playback at 704 pixel width.

    if (avctx->width == 720 &&

        (avctx->height == 480 || avctx->height == 576)) {

        av_reduce(&dw, &dh,

                  ctx->init_encode_params.darWidth * 44,

                  ctx->init_encode_params.darHeight * 45,

                  1024 * 1024);

        ctx->init_encode_params.darHeight = dh;

        ctx->init_encode_params.darWidth = dw;

    }



    ctx->init_encode_params.frameRateNum = avctx->time_base.den;

    ctx->init_encode_params.frameRateDen = avctx->time_base.num * avctx->ticks_per_frame;



    num_mbs = ((avctx->width + 15) >> 4) * ((avctx->height + 15) >> 4);

    ctx->max_surface_count = (num_mbs >= 8160) ? 32 : 48;



    if (ctx->buffer_delay >= ctx->max_surface_count)

        ctx->buffer_delay = ctx->max_surface_count - 1;



    ctx->init_encode_params.enableEncodeAsync = 0;

    ctx->init_encode_params.enablePTD = 1;



    ctx->init_encode_params.presetGUID = encoder_preset;



    ctx->init_encode_params.encodeConfig = &ctx->encode_config;

    memcpy(&ctx->encode_config, &preset_config.presetCfg, sizeof(ctx->encode_config));

    ctx->encode_config.version = NV_ENC_CONFIG_VER;



    if (avctx->refs >= 0) {

        /* 0 means "let the hardware decide" */

        switch (avctx->codec->id) {

        case AV_CODEC_ID_H264:

            ctx->encode_config.encodeCodecConfig.h264Config.maxNumRefFrames = avctx->refs;

            break;

        case AV_CODEC_ID_H265:

            ctx->encode_config.encodeCodecConfig.hevcConfig.maxNumRefFramesInDPB = avctx->refs;

            break;

        /* Earlier switch/case will return if unknown codec is passed. */

        }

    }



    if (avctx->gop_size > 0) {

        if (avctx->max_b_frames >= 0) {

            /* 0 is intra-only, 1 is I/P only, 2 is one B Frame, 3 two B frames, and so on. */

            ctx->encode_config.frameIntervalP = avctx->max_b_frames + 1;

        }



        ctx->encode_config.gopLength = avctx->gop_size;

        switch (avctx->codec->id) {

        case AV_CODEC_ID_H264:

            ctx->encode_config.encodeCodecConfig.h264Config.idrPeriod = avctx->gop_size;

            break;

        case AV_CODEC_ID_H265:

            ctx->encode_config.encodeCodecConfig.hevcConfig.idrPeriod = avctx->gop_size;

            break;

        /* Earlier switch/case will return if unknown codec is passed. */

        }

    } else if (avctx->gop_size == 0) {

        ctx->encode_config.frameIntervalP = 0;

        ctx->encode_config.gopLength = 1;

        switch (avctx->codec->id) {

        case AV_CODEC_ID_H264:

            ctx->encode_config.encodeCodecConfig.h264Config.idrPeriod = 1;

            break;

        case AV_CODEC_ID_H265:

            ctx->encode_config.encodeCodecConfig.hevcConfig.idrPeriod = 1;

            break;

        /* Earlier switch/case will return if unknown codec is passed. */

        }

    }



    /* when there're b frames, set dts offset */

    if (ctx->encode_config.frameIntervalP >= 2)

        ctx->last_dts = -2;



    if (avctx->bit_rate > 0) {

        ctx->encode_config.rcParams.averageBitRate = avctx->bit_rate;

    } else if (ctx->encode_config.rcParams.averageBitRate > 0) {

        ctx->encode_config.rcParams.maxBitRate = ctx->encode_config.rcParams.averageBitRate;

    }



    if (avctx->rc_max_rate > 0)

        ctx->encode_config.rcParams.maxBitRate = avctx->rc_max_rate;



    if (lossless) {

        if (avctx->codec->id == AV_CODEC_ID_H264)

            ctx->encode_config.encodeCodecConfig.h264Config.qpPrimeYZeroTransformBypassFlag = 1;



        ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

        ctx->encode_config.rcParams.constQP.qpInterB = 0;

        ctx->encode_config.rcParams.constQP.qpInterP = 0;

        ctx->encode_config.rcParams.constQP.qpIntra = 0;



        avctx->qmin = -1;

        avctx->qmax = -1;

    } else if (ctx->cbr) {

        if (!ctx->twopass) {

            ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CBR;

        } else {

            ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_QUALITY;



            if (avctx->codec->id == AV_CODEC_ID_H264) {

                ctx->encode_config.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;

                ctx->encode_config.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;

            }

        }



        if (avctx->codec->id == AV_CODEC_ID_H264) {

            ctx->encode_config.encodeCodecConfig.h264Config.outputBufferingPeriodSEI = 1;

            ctx->encode_config.encodeCodecConfig.h264Config.outputPictureTimingSEI = 1;

        } else if(avctx->codec->id == AV_CODEC_ID_H265) {

            ctx->encode_config.encodeCodecConfig.hevcConfig.outputBufferingPeriodSEI = 1;

            ctx->encode_config.encodeCodecConfig.hevcConfig.outputPictureTimingSEI = 1;

        }

    } else if (avctx->global_quality > 0) {

        ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_CONSTQP;

        ctx->encode_config.rcParams.constQP.qpInterB = avctx->global_quality;

        ctx->encode_config.rcParams.constQP.qpInterP = avctx->global_quality;

        ctx->encode_config.rcParams.constQP.qpIntra = avctx->global_quality;



        avctx->qmin = -1;

        avctx->qmax = -1;

    } else {

        if (avctx->qmin >= 0 && avctx->qmax >= 0) {

            ctx->encode_config.rcParams.enableMinQP = 1;

            ctx->encode_config.rcParams.enableMaxQP = 1;



            ctx->encode_config.rcParams.minQP.qpInterB = avctx->qmin;

            ctx->encode_config.rcParams.minQP.qpInterP = avctx->qmin;

            ctx->encode_config.rcParams.minQP.qpIntra = avctx->qmin;



            ctx->encode_config.rcParams.maxQP.qpInterB = avctx->qmax;

            ctx->encode_config.rcParams.maxQP.qpInterP = avctx->qmax;

            ctx->encode_config.rcParams.maxQP.qpIntra = avctx->qmax;



            qp_inter_p = (avctx->qmax + 3 * avctx->qmin) / 4; // biased towards Qmin



            if (ctx->twopass) {

                ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_VBR;

                if (avctx->codec->id == AV_CODEC_ID_H264) {

                    ctx->encode_config.encodeCodecConfig.h264Config.adaptiveTransformMode = NV_ENC_H264_ADAPTIVE_TRANSFORM_ENABLE;

                    ctx->encode_config.encodeCodecConfig.h264Config.fmoMode = NV_ENC_H264_FMO_DISABLE;

                }

            } else {

                ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR_MINQP;

            }

        } else {

            qp_inter_p = 26; // default to 26



            if (ctx->twopass) {

                ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_2_PASS_VBR;

            } else {

                ctx->encode_config.rcParams.rateControlMode = NV_ENC_PARAMS_RC_VBR;

            }

        }



        ctx->encode_config.rcParams.enableInitialRCQP = 1;

        ctx->encode_config.rcParams.initialRCQP.qpInterP  = qp_inter_p;



        if(avctx->i_quant_factor != 0.0 && avctx->b_quant_factor != 0.0) {

            ctx->encode_config.rcParams.initialRCQP.qpIntra = av_clip(

                qp_inter_p * fabs(avctx->i_quant_factor) + avctx->i_quant_offset, 0, 51);

            ctx->encode_config.rcParams.initialRCQP.qpInterB = av_clip(

                qp_inter_p * fabs(avctx->b_quant_factor) + avctx->b_quant_offset, 0, 51);

        } else {

            ctx->encode_config.rcParams.initialRCQP.qpIntra = qp_inter_p;

            ctx->encode_config.rcParams.initialRCQP.qpInterB = qp_inter_p;

        }

    }



    if (avctx->rc_buffer_size > 0) {

        ctx->encode_config.rcParams.vbvBufferSize = avctx->rc_buffer_size;

    } else if (ctx->encode_config.rcParams.averageBitRate > 0) {

        ctx->encode_config.rcParams.vbvBufferSize = 2 * ctx->encode_config.rcParams.averageBitRate;

    }



    if (avctx->flags & AV_CODEC_FLAG_INTERLACED_DCT) {

        ctx->encode_config.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FIELD;

    } else {

        ctx->encode_config.frameFieldMode = NV_ENC_PARAMS_FRAME_FIELD_MODE_FRAME;

    }



    switch (avctx->codec->id) {

    case AV_CODEC_ID_H264:

        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourMatrix = avctx->colorspace;

        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourPrimaries = avctx->color_primaries;

        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.transferCharacteristics = avctx->color_trc;

        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag = (avctx->color_range == AVCOL_RANGE_JPEG

            || avctx->pix_fmt == AV_PIX_FMT_YUVJ420P || avctx->pix_fmt == AV_PIX_FMT_YUVJ422P || avctx->pix_fmt == AV_PIX_FMT_YUVJ444P);



        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag =

            (avctx->colorspace != 2 || avctx->color_primaries != 2 || avctx->color_trc != 2);



        ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoSignalTypePresentFlag =

            (ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.colourDescriptionPresentFlag

            || ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoFormat != 5

            || ctx->encode_config.encodeCodecConfig.h264Config.h264VUIParameters.videoFullRangeFlag != 0);



        ctx->encode_config.encodeCodecConfig.h264Config.sliceMode = 3;

        ctx->encode_config.encodeCodecConfig.h264Config.sliceModeData = 1;



        ctx->encode_config.encodeCodecConfig.h264Config.disableSPSPPS = (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) ? 1 : 0;

        ctx->encode_config.encodeCodecConfig.h264Config.repeatSPSPPS = (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) ? 0 : 1;



        ctx->encode_config.encodeCodecConfig.h264Config.outputAUD = 1;



        if (!ctx->profile && !lossless) {

            switch (avctx->profile) {

            case FF_PROFILE_H264_HIGH_444_PREDICTIVE:

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;

                break;

            case FF_PROFILE_H264_BASELINE:

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;

                break;

            case FF_PROFILE_H264_MAIN:

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID;

                break;

            case FF_PROFILE_H264_HIGH:

            case FF_PROFILE_UNKNOWN:

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;

                break;

            default:

                av_log(avctx, AV_LOG_WARNING, "Unsupported profile requested, falling back to high\n");

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;

                break;

            }

        } else if(!lossless) {

            if (!strcmp(ctx->profile, "high")) {

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_GUID;

                avctx->profile = FF_PROFILE_H264_HIGH;

            } else if (!strcmp(ctx->profile, "main")) {

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_MAIN_GUID;

                avctx->profile = FF_PROFILE_H264_MAIN;

            } else if (!strcmp(ctx->profile, "baseline")) {

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_BASELINE_GUID;

                avctx->profile = FF_PROFILE_H264_BASELINE;

            } else if (!strcmp(ctx->profile, "high444p")) {

                ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;

                avctx->profile = FF_PROFILE_H264_HIGH_444_PREDICTIVE;

            } else {

                av_log(avctx, AV_LOG_FATAL, "Profile \"%s\" is unknown! Supported profiles: high, main, baseline\n", ctx->profile);

                res = AVERROR(EINVAL);

                goto error;

            }

        }



        // force setting profile as high444p if input is AV_PIX_FMT_YUV444P

        if (avctx->pix_fmt == AV_PIX_FMT_YUV444P) {

            ctx->encode_config.profileGUID = NV_ENC_H264_PROFILE_HIGH_444_GUID;

            avctx->profile = FF_PROFILE_H264_HIGH_444_PREDICTIVE;

        }



        ctx->encode_config.encodeCodecConfig.h264Config.chromaFormatIDC = avctx->profile == FF_PROFILE_H264_HIGH_444_PREDICTIVE ? 3 : 1;



        if (ctx->level) {

            res = input_string_to_uint32(avctx, nvenc_h264_level_pairs, ctx->level, &ctx->encode_config.encodeCodecConfig.h264Config.level);



            if (res) {

                av_log(avctx, AV_LOG_FATAL, "Level \"%s\" is unknown! Supported levels: auto, 1, 1b, 1.1, 1.2, 1.3, 2, 2.1, 2.2, 3, 3.1, 3.2, 4, 4.1, 4.2, 5, 5.1\n", ctx->level);

                goto error;

            }

        } else {

            ctx->encode_config.encodeCodecConfig.h264Config.level = NV_ENC_LEVEL_AUTOSELECT;

        }



        break;

    case AV_CODEC_ID_H265:

        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.colourMatrix = avctx->colorspace;

        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.colourPrimaries = avctx->color_primaries;

        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.transferCharacteristics = avctx->color_trc;

        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFullRangeFlag = (avctx->color_range == AVCOL_RANGE_JPEG

            || avctx->pix_fmt == AV_PIX_FMT_YUVJ420P || avctx->pix_fmt == AV_PIX_FMT_YUVJ422P || avctx->pix_fmt == AV_PIX_FMT_YUVJ444P);



        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag =

            (avctx->colorspace != 2 || avctx->color_primaries != 2 || avctx->color_trc != 2);



        ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.videoSignalTypePresentFlag =

            (ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.colourDescriptionPresentFlag

            || ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFormat != 5

            || ctx->encode_config.encodeCodecConfig.hevcConfig.hevcVUIParameters.videoFullRangeFlag != 0);



        ctx->encode_config.encodeCodecConfig.hevcConfig.sliceMode = 3;

        ctx->encode_config.encodeCodecConfig.hevcConfig.sliceModeData = 1;



        ctx->encode_config.encodeCodecConfig.hevcConfig.disableSPSPPS = (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) ? 1 : 0;

        ctx->encode_config.encodeCodecConfig.hevcConfig.repeatSPSPPS = (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) ? 0 : 1;



        ctx->encode_config.encodeCodecConfig.hevcConfig.outputAUD = 1;



        /* No other profile is supported in the current SDK version 5 */

        ctx->encode_config.profileGUID = NV_ENC_HEVC_PROFILE_MAIN_GUID;

        avctx->profile = FF_PROFILE_HEVC_MAIN;



        if (ctx->level) {

            res = input_string_to_uint32(avctx, nvenc_hevc_level_pairs, ctx->level, &ctx->encode_config.encodeCodecConfig.hevcConfig.level);



            if (res) {

                av_log(avctx, AV_LOG_FATAL, "Level \"%s\" is unknown! Supported levels: auto, 1, 2, 2.1, 3, 3.1, 4, 4.1, 5, 5.1, 5.2, 6, 6.1, 6.2\n", ctx->level);

                goto error;

            }

        } else {

            ctx->encode_config.encodeCodecConfig.hevcConfig.level = NV_ENC_LEVEL_AUTOSELECT;

        }



        if (ctx->tier) {

            if (!strcmp(ctx->tier, "main")) {

                ctx->encode_config.encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_MAIN;

            } else if (!strcmp(ctx->tier, "high")) {

                ctx->encode_config.encodeCodecConfig.hevcConfig.tier = NV_ENC_TIER_HEVC_HIGH;

            } else {

                av_log(avctx, AV_LOG_FATAL, "Tier \"%s\" is unknown! Supported tiers: main, high\n", ctx->tier);

                res = AVERROR(EINVAL);

                goto error;

            }

        }



        break;

    /* Earlier switch/case will return if unknown codec is passed. */

    }



    nv_status = p_nvenc->nvEncInitializeEncoder(ctx->nvencoder, &ctx->init_encode_params);

    if (nv_status != NV_ENC_SUCCESS) {

        av_log(avctx, AV_LOG_FATAL, "InitializeEncoder failed: 0x%x\n", (int)nv_status);

        res = AVERROR_EXTERNAL;

        goto error;

    }



    ctx->input_surfaces = av_malloc(ctx->max_surface_count * sizeof(*ctx->input_surfaces));



    if (!ctx->input_surfaces) {

        res = AVERROR(ENOMEM);

        goto error;

    }



    ctx->output_surfaces = av_malloc(ctx->max_surface_count * sizeof(*ctx->output_surfaces));



    if (!ctx->output_surfaces) {

        res = AVERROR(ENOMEM);

        goto error;

    }



    for (surfaceCount = 0; surfaceCount < ctx->max_surface_count; ++surfaceCount) {

        NV_ENC_CREATE_INPUT_BUFFER allocSurf = { 0 };

        NV_ENC_CREATE_BITSTREAM_BUFFER allocOut = { 0 };

        allocSurf.version = NV_ENC_CREATE_INPUT_BUFFER_VER;

        allocOut.version = NV_ENC_CREATE_BITSTREAM_BUFFER_VER;



        allocSurf.width = (avctx->width + 31) & ~31;

        allocSurf.height = (avctx->height + 31) & ~31;



        allocSurf.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;



        switch (avctx->pix_fmt) {

        case AV_PIX_FMT_YUV420P:

            allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_YV12_PL;

            break;



        case AV_PIX_FMT_NV12:

            allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_NV12_PL;

            break;



        case AV_PIX_FMT_YUV444P:

            allocSurf.bufferFmt = NV_ENC_BUFFER_FORMAT_YUV444_PL;

            break;



        default:

            av_log(avctx, AV_LOG_FATAL, "Invalid input pixel format\n");

            res = AVERROR(EINVAL);

            goto error;

        }



        nv_status = p_nvenc->nvEncCreateInputBuffer(ctx->nvencoder, &allocSurf);

        if (nv_status != NV_ENC_SUCCESS) {

            av_log(avctx, AV_LOG_FATAL, "CreateInputBuffer failed\n");

            res = AVERROR_EXTERNAL;

            goto error;

        }



        ctx->input_surfaces[surfaceCount].lockCount = 0;

        ctx->input_surfaces[surfaceCount].input_surface = allocSurf.inputBuffer;

        ctx->input_surfaces[surfaceCount].format = allocSurf.bufferFmt;

        ctx->input_surfaces[surfaceCount].width = allocSurf.width;

        ctx->input_surfaces[surfaceCount].height = allocSurf.height;



        /* 1MB is large enough to hold most output frames. NVENC increases this automaticaly if it's not enough. */

        allocOut.size = 1024 * 1024;



        allocOut.memoryHeap = NV_ENC_MEMORY_HEAP_SYSMEM_CACHED;



        nv_status = p_nvenc->nvEncCreateBitstreamBuffer(ctx->nvencoder, &allocOut);

        if (nv_status != NV_ENC_SUCCESS) {

            av_log(avctx, AV_LOG_FATAL, "CreateBitstreamBuffer failed\n");

            ctx->output_surfaces[surfaceCount++].output_surface = NULL;

            res = AVERROR_EXTERNAL;

            goto error;

        }



        ctx->output_surfaces[surfaceCount].output_surface = allocOut.bitstreamBuffer;

        ctx->output_surfaces[surfaceCount].size = allocOut.size;

        ctx->output_surfaces[surfaceCount].busy = 0;

    }



    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        uint32_t outSize = 0;

        char tmpHeader[256];

        NV_ENC_SEQUENCE_PARAM_PAYLOAD payload = { 0 };

        payload.version = NV_ENC_SEQUENCE_PARAM_PAYLOAD_VER;



        payload.spsppsBuffer = tmpHeader;

        payload.inBufferSize = sizeof(tmpHeader);

        payload.outSPSPPSPayloadSize = &outSize;



        nv_status = p_nvenc->nvEncGetSequenceParams(ctx->nvencoder, &payload);

        if (nv_status != NV_ENC_SUCCESS) {

            av_log(avctx, AV_LOG_FATAL, "GetSequenceParams failed\n");

            goto error;

        }



        avctx->extradata_size = outSize;

        avctx->extradata = av_mallocz(outSize + AV_INPUT_BUFFER_PADDING_SIZE);



        if (!avctx->extradata) {

            res = AVERROR(ENOMEM);

            goto error;

        }



        memcpy(avctx->extradata, tmpHeader, outSize);

    }



    if (ctx->encode_config.frameIntervalP > 1)

        avctx->has_b_frames = 2;



    if (ctx->encode_config.rcParams.averageBitRate > 0)

        avctx->bit_rate = ctx->encode_config.rcParams.averageBitRate;



    cpb_props = ff_add_cpb_side_data(avctx);

    if (!cpb_props)

        return AVERROR(ENOMEM);

    cpb_props->max_bitrate = ctx->encode_config.rcParams.maxBitRate;

    cpb_props->avg_bitrate = avctx->bit_rate;

    cpb_props->buffer_size = ctx->encode_config.rcParams.vbvBufferSize;



    return 0;



error:



    for (i = 0; i < surfaceCount; ++i) {

        p_nvenc->nvEncDestroyInputBuffer(ctx->nvencoder, ctx->input_surfaces[i].input_surface);

        if (ctx->output_surfaces[i].output_surface)

            p_nvenc->nvEncDestroyBitstreamBuffer(ctx->nvencoder, ctx->output_surfaces[i].output_surface);

    }



    if (ctx->nvencoder)

        p_nvenc->nvEncDestroyEncoder(ctx->nvencoder);



    if (ctx->cu_context)

        dl_fn->cu_ctx_destroy(ctx->cu_context);



    nvenc_unload_nvenc(avctx);



    ctx->nvencoder = NULL;

    ctx->cu_context = NULL;



    return res;

}
