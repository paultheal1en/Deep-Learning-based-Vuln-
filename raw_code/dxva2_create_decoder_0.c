static int dxva2_create_decoder(AVCodecContext *s)

{

    InputStream  *ist = s->opaque;

    int loglevel = (ist->hwaccel_id == HWACCEL_AUTO) ? AV_LOG_VERBOSE : AV_LOG_ERROR;

    DXVA2Context *ctx = ist->hwaccel_ctx;

    struct dxva_context *dxva_ctx = s->hwaccel_context;

    GUID *guid_list = NULL;

    unsigned guid_count = 0, i, j;

    GUID device_guid = GUID_NULL;

    const D3DFORMAT surface_format = (s->sw_pix_fmt == AV_PIX_FMT_YUV420P10) ? MKTAG('P','0','1','0') : MKTAG('N','V','1','2');

    D3DFORMAT target_format = 0;

    DXVA2_VideoDesc desc = { 0 };

    DXVA2_ConfigPictureDecode config;

    HRESULT hr;

    int surface_alignment, num_surfaces;

    int ret;



    AVDXVA2FramesContext *frames_hwctx;

    AVHWFramesContext *frames_ctx;



    hr = IDirectXVideoDecoderService_GetDecoderDeviceGuids(ctx->decoder_service, &guid_count, &guid_list);

    if (FAILED(hr)) {

        av_log(NULL, loglevel, "Failed to retrieve decoder device GUIDs\n");

        goto fail;

    }



    for (i = 0; dxva2_modes[i].guid; i++) {

        D3DFORMAT *target_list = NULL;

        unsigned target_count = 0;

        const dxva2_mode *mode = &dxva2_modes[i];

        if (mode->codec != s->codec_id)

            continue;



        for (j = 0; j < guid_count; j++) {

            if (IsEqualGUID(mode->guid, &guid_list[j]))

                break;

        }

        if (j == guid_count)

            continue;



        hr = IDirectXVideoDecoderService_GetDecoderRenderTargets(ctx->decoder_service, mode->guid, &target_count, &target_list);

        if (FAILED(hr)) {

            continue;

        }

        for (j = 0; j < target_count; j++) {

            const D3DFORMAT format = target_list[j];

            if (format == surface_format) {

                target_format = format;

                break;

            }

        }

        CoTaskMemFree(target_list);

        if (target_format) {

            device_guid = *mode->guid;

            break;

        }

    }

    CoTaskMemFree(guid_list);



    if (IsEqualGUID(&device_guid, &GUID_NULL)) {

        av_log(NULL, loglevel, "No decoder device for codec found\n");

        goto fail;

    }



    desc.SampleWidth  = s->coded_width;

    desc.SampleHeight = s->coded_height;

    desc.Format       = target_format;



    ret = dxva2_get_decoder_configuration(s, &device_guid, &desc, &config);

    if (ret < 0) {

        goto fail;

    }



    /* decoding MPEG-2 requires additional alignment on some Intel GPUs,

       but it causes issues for H.264 on certain AMD GPUs..... */

    if (s->codec_id == AV_CODEC_ID_MPEG2VIDEO)

        surface_alignment = 32;

    /* the HEVC DXVA2 spec asks for 128 pixel aligned surfaces to ensure

       all coding features have enough room to work with */

    else if  (s->codec_id == AV_CODEC_ID_HEVC)

        surface_alignment = 128;

    else

        surface_alignment = 16;



    /* 4 base work surfaces */

    num_surfaces = 4;



    /* add surfaces based on number of possible refs */

    if (s->codec_id == AV_CODEC_ID_H264 || s->codec_id == AV_CODEC_ID_HEVC)

        num_surfaces += 16;

    else if (s->codec_id == AV_CODEC_ID_VP9)

        num_surfaces += 8;

    else

        num_surfaces += 2;



    /* add extra surfaces for frame threading */

    if (s->active_thread_type & FF_THREAD_FRAME)

        num_surfaces += s->thread_count;



    ctx->hw_frames_ctx = av_hwframe_ctx_alloc(ctx->hw_device_ctx);

    if (!ctx->hw_frames_ctx)

        goto fail;

    frames_ctx   = (AVHWFramesContext*)ctx->hw_frames_ctx->data;

    frames_hwctx = frames_ctx->hwctx;



    frames_ctx->format            = AV_PIX_FMT_DXVA2_VLD;

    frames_ctx->sw_format         = (target_format == MKTAG('P','0','1','0') ? AV_PIX_FMT_P010 : AV_PIX_FMT_NV12);

    frames_ctx->width             = FFALIGN(s->coded_width, surface_alignment);

    frames_ctx->height            = FFALIGN(s->coded_height, surface_alignment);

    frames_ctx->initial_pool_size = num_surfaces;



    frames_hwctx->surface_type = DXVA2_VideoDecoderRenderTarget;



    ret = av_hwframe_ctx_init(ctx->hw_frames_ctx);

    if (ret < 0) {

        av_log(NULL, loglevel, "Failed to initialize the HW frames context\n");

        goto fail;

    }



    hr = IDirectXVideoDecoderService_CreateVideoDecoder(ctx->decoder_service, &device_guid,

                                                        &desc, &config, frames_hwctx->surfaces,

                                                        frames_hwctx->nb_surfaces, &frames_hwctx->decoder_to_release);

    if (FAILED(hr)) {

        av_log(NULL, loglevel, "Failed to create DXVA2 video decoder\n");

        goto fail;

    }



    ctx->decoder_guid   = device_guid;

    ctx->decoder_config = config;



    dxva_ctx->cfg           = &ctx->decoder_config;

    dxva_ctx->decoder       = frames_hwctx->decoder_to_release;

    dxva_ctx->surface       = frames_hwctx->surfaces;

    dxva_ctx->surface_count = frames_hwctx->nb_surfaces;



    if (IsEqualGUID(&ctx->decoder_guid, &DXVADDI_Intel_ModeH264_E))

        dxva_ctx->workaround |= FF_DXVA2_WORKAROUND_INTEL_CLEARVIDEO;



    return 0;

fail:

    av_buffer_unref(&ctx->hw_frames_ctx);

    return AVERROR(EINVAL);

}
