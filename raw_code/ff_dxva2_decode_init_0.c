int ff_dxva2_decode_init(AVCodecContext *avctx)

{

    FFDXVASharedContext *sctx = DXVA_SHARED_CONTEXT(avctx);

    AVHWFramesContext *frames_ctx = NULL;

    int ret = 0;



    // Old API.

    if (avctx->hwaccel_context)

        return 0;



    // (avctx->pix_fmt is not updated yet at this point)

    sctx->pix_fmt = avctx->hwaccel->pix_fmt;



    if (avctx->codec_id == AV_CODEC_ID_H264 &&

        (avctx->profile & ~FF_PROFILE_H264_CONSTRAINED) > FF_PROFILE_H264_HIGH) {

        av_log(avctx, AV_LOG_VERBOSE, "Unsupported H.264 profile for DXVA HWAccel: %d\n",avctx->profile);

        return AVERROR(ENOTSUP);

    }



    if (avctx->codec_id == AV_CODEC_ID_HEVC &&

        avctx->profile != FF_PROFILE_HEVC_MAIN && avctx->profile != FF_PROFILE_HEVC_MAIN_10) {

        av_log(avctx, AV_LOG_VERBOSE, "Unsupported HEVC profile for DXVA HWAccel: %d\n", avctx->profile);

        return AVERROR(ENOTSUP);

    }



    if (!avctx->hw_frames_ctx && !avctx->hw_device_ctx) {

        av_log(avctx, AV_LOG_ERROR, "Either a hw_frames_ctx or a hw_device_ctx needs to be set for hardware decoding.\n");

        return AVERROR(EINVAL);

    }



    if (avctx->hw_frames_ctx) {

        frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;

    } else {

        avctx->hw_frames_ctx = av_hwframe_ctx_alloc(avctx->hw_device_ctx);

        if (!avctx->hw_frames_ctx)

            return AVERROR(ENOMEM);



        frames_ctx = (AVHWFramesContext*)avctx->hw_frames_ctx->data;



        dxva_adjust_hwframes(avctx, frames_ctx);



        ret = av_hwframe_ctx_init(avctx->hw_frames_ctx);

        if (ret < 0)

            goto fail;

    }



    sctx->device_ctx = frames_ctx->device_ctx;



    if (frames_ctx->format != sctx->pix_fmt ||

        !((sctx->pix_fmt == AV_PIX_FMT_D3D11 && CONFIG_D3D11VA) ||

          (sctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD && CONFIG_DXVA2))) {

        av_log(avctx, AV_LOG_ERROR, "Invalid pixfmt for hwaccel!\n");

        ret = AVERROR(EINVAL);

        goto fail;

    }



#if CONFIG_D3D11VA

    if (sctx->pix_fmt == AV_PIX_FMT_D3D11) {

        AVD3D11VADeviceContext *device_hwctx = frames_ctx->device_ctx->hwctx;

        AVD3D11VAContext *d3d11_ctx = &sctx->ctx.d3d11va;

        HRESULT hr;



        ff_dxva2_lock(avctx);

        ret = d3d11va_create_decoder(avctx);

        ff_dxva2_unlock(avctx);

        if (ret < 0)

            goto fail;



        d3d11_ctx->decoder       = sctx->d3d11_decoder;

        d3d11_ctx->video_context = device_hwctx->video_context;

        d3d11_ctx->cfg           = &sctx->d3d11_config;

        d3d11_ctx->surface_count = sctx->nb_d3d11_views;

        d3d11_ctx->surface       = sctx->d3d11_views;

        d3d11_ctx->workaround    = sctx->workaround;

        d3d11_ctx->context_mutex = INVALID_HANDLE_VALUE;

    }

#endif



#if CONFIG_DXVA2

    if (sctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

        AVDXVA2FramesContext *frames_hwctx = frames_ctx->hwctx;

        struct dxva_context *dxva_ctx = &sctx->ctx.dxva2;



        ff_dxva2_lock(avctx);

        ret = dxva2_create_decoder(avctx);

        ff_dxva2_unlock(avctx);

        if (ret < 0)

            goto fail;



        dxva_ctx->decoder       = sctx->dxva2_decoder;

        dxva_ctx->cfg           = &sctx->dxva2_config;

        dxva_ctx->surface       = frames_hwctx->surfaces;

        dxva_ctx->surface_count = frames_hwctx->nb_surfaces;

        dxva_ctx->workaround    = sctx->workaround;

    }

#endif



    return 0;



fail:

    ff_dxva2_decode_uninit(avctx);

    return ret;

}
