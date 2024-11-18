static av_cold int vtenc_init(AVCodecContext *avctx)

{

    CFMutableDictionaryRef enc_info;

    CFMutableDictionaryRef pixel_buffer_info;

    CMVideoCodecType       codec_type;

    VTEncContext           *vtctx = avctx->priv_data;

    CFStringRef            profile_level;

    CFBooleanRef           has_b_frames_cfbool;

    CFNumberRef            gamma_level = NULL;

    int                    status;



    pthread_once(&once_ctrl, loadVTEncSymbols);



    codec_type = get_cm_codec_type(avctx->codec_id);

    if (!codec_type) {

        av_log(avctx, AV_LOG_ERROR, "Error: no mapping for AVCodecID %d\n", avctx->codec_id);

        return AVERROR(EINVAL);

    }



    vtctx->has_b_frames = avctx->max_b_frames > 0;

    if(vtctx->has_b_frames && vtctx->profile == H264_PROF_BASELINE){

        av_log(avctx, AV_LOG_WARNING, "Cannot use B-frames with baseline profile. Output will not contain B-frames.\n");

        vtctx->has_b_frames = false;

    }



    if (vtctx->entropy == VT_CABAC && vtctx->profile == H264_PROF_BASELINE) {

        av_log(avctx, AV_LOG_WARNING, "CABAC entropy requires 'main' or 'high' profile, but baseline was requested. Encode will not use CABAC entropy.\n");

        vtctx->entropy = VT_ENTROPY_NOT_SET;

    }



    if (!get_vt_profile_level(avctx, &profile_level)) return AVERROR(EINVAL);



    vtctx->session = NULL;



    enc_info = CFDictionaryCreateMutable(

        kCFAllocatorDefault,

        20,

        &kCFCopyStringDictionaryKeyCallBacks,

        &kCFTypeDictionaryValueCallBacks

    );



    if (!enc_info) return AVERROR(ENOMEM);



#if !TARGET_OS_IPHONE

    if (!vtctx->allow_sw) {

        CFDictionarySetValue(enc_info,

                             compat_keys.kVTVideoEncoderSpecification_RequireHardwareAcceleratedVideoEncoder,

                             kCFBooleanTrue);

    } else {

        CFDictionarySetValue(enc_info,

                             compat_keys.kVTVideoEncoderSpecification_EnableHardwareAcceleratedVideoEncoder,

                             kCFBooleanTrue);

    }

#endif



    if (avctx->pix_fmt != AV_PIX_FMT_VIDEOTOOLBOX) {

        status = create_cv_pixel_buffer_info(avctx, &pixel_buffer_info);

        if (status)

            goto init_cleanup;

    } else {

        pixel_buffer_info = NULL;

    }



    pthread_mutex_init(&vtctx->lock, NULL);

    pthread_cond_init(&vtctx->cv_sample_sent, NULL);

    vtctx->dts_delta = vtctx->has_b_frames ? -1 : 0;



    get_cv_transfer_function(avctx, &vtctx->transfer_function, &gamma_level);

    get_cv_ycbcr_matrix(avctx, &vtctx->ycbcr_matrix);

    get_cv_color_primaries(avctx, &vtctx->color_primaries);





    if (avctx->flags & AV_CODEC_FLAG_GLOBAL_HEADER) {

        status = vtenc_populate_extradata(avctx,

                                          codec_type,

                                          profile_level,

                                          gamma_level,

                                          enc_info,

                                          pixel_buffer_info);

        if (status)

            goto init_cleanup;

    }



    status = vtenc_create_encoder(avctx,

                                  codec_type,

                                  profile_level,

                                  gamma_level,

                                  enc_info,

                                  pixel_buffer_info,

                                  &vtctx->session);



    if (status < 0)

        goto init_cleanup;



    status = VTSessionCopyProperty(vtctx->session,

                                   kVTCompressionPropertyKey_AllowFrameReordering,

                                   kCFAllocatorDefault,

                                   &has_b_frames_cfbool);



    if (!status) {

        //Some devices don't output B-frames for main profile, even if requested.

        vtctx->has_b_frames = CFBooleanGetValue(has_b_frames_cfbool);

        CFRelease(has_b_frames_cfbool);

    }

    avctx->has_b_frames = vtctx->has_b_frames;



init_cleanup:

    if (gamma_level)

        CFRelease(gamma_level);



    if (pixel_buffer_info)

        CFRelease(pixel_buffer_info);



    CFRelease(enc_info);



    return status;

}
