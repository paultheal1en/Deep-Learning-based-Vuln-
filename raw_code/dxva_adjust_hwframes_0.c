static void dxva_adjust_hwframes(AVCodecContext *avctx, AVHWFramesContext *frames_ctx)

{

    FFDXVASharedContext *sctx = DXVA_SHARED_CONTEXT(avctx);

    int surface_alignment, num_surfaces;



    frames_ctx->format = sctx->pix_fmt;



    /* decoding MPEG-2 requires additional alignment on some Intel GPUs,

    but it causes issues for H.264 on certain AMD GPUs..... */

    if (avctx->codec_id == AV_CODEC_ID_MPEG2VIDEO)

        surface_alignment = 32;

    /* the HEVC DXVA2 spec asks for 128 pixel aligned surfaces to ensure

    all coding features have enough room to work with */

    else if (avctx->codec_id == AV_CODEC_ID_HEVC)

        surface_alignment = 128;

    else

        surface_alignment = 16;



    /* 4 base work surfaces */

    num_surfaces = 4;



    /* add surfaces based on number of possible refs */

    if (avctx->codec_id == AV_CODEC_ID_H264 || avctx->codec_id == AV_CODEC_ID_HEVC)

        num_surfaces += 16;

    else

        num_surfaces += 2;



    /* add extra surfaces for frame threading */

    if (avctx->active_thread_type & FF_THREAD_FRAME)

        num_surfaces += avctx->thread_count;



    frames_ctx->sw_format = avctx->sw_pix_fmt == AV_PIX_FMT_YUV420P10 ?

                            AV_PIX_FMT_P010 : AV_PIX_FMT_NV12;

    frames_ctx->width = FFALIGN(avctx->coded_width, surface_alignment);

    frames_ctx->height = FFALIGN(avctx->coded_height, surface_alignment);

    frames_ctx->initial_pool_size = num_surfaces;





#if CONFIG_DXVA2

    if (frames_ctx->format == AV_PIX_FMT_DXVA2_VLD) {

        AVDXVA2FramesContext *frames_hwctx = frames_ctx->hwctx;



        frames_hwctx->surface_type = DXVA2_VideoDecoderRenderTarget;

    }

#endif



#if CONFIG_D3D11VA

    if (frames_ctx->format == AV_PIX_FMT_D3D11) {

        AVD3D11VAFramesContext *frames_hwctx = frames_ctx->hwctx;



        frames_hwctx->BindFlags |= D3D11_BIND_DECODER;

    }

#endif

}
