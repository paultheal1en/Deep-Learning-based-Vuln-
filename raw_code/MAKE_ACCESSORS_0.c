MAKE_ACCESSORS(AVVDPAUContext, vdpau_hwaccel, AVVDPAU_Render2, render2)



int ff_vdpau_common_init(AVCodecContext *avctx, VdpDecoderProfile profile,

                         int level)

{

    VDPAUHWContext *hwctx = avctx->hwaccel_context;

    VDPAUContext *vdctx = avctx->internal->hwaccel_priv_data;

    VdpVideoSurfaceQueryCapabilities *surface_query_caps;

    VdpDecoderQueryCapabilities *decoder_query_caps;

    VdpDecoderCreate *create;

    void *func;

    VdpStatus status;

    VdpBool supported;

    uint32_t max_level, max_mb, max_width, max_height;

    /* See vdpau/vdpau.h for alignment constraints. */

    uint32_t width  = (avctx->coded_width + 1) & ~1;

    uint32_t height = (avctx->coded_height + 3) & ~3;



    vdctx->width            = UINT32_MAX;

    vdctx->height           = UINT32_MAX;

    hwctx->reset            = 0;



    if (!hwctx) {

        vdctx->device  = VDP_INVALID_HANDLE;

        av_log(avctx, AV_LOG_WARNING, "hwaccel_context has not been setup by the user application, cannot initialize\n");

        return 0;

    }



    if (hwctx->context.decoder != VDP_INVALID_HANDLE) {

        vdctx->decoder = hwctx->context.decoder;

        vdctx->render  = hwctx->context.render;

        vdctx->device  = VDP_INVALID_HANDLE;

        return 0; /* Decoder created by user */

    }



    vdctx->device           = hwctx->device;

    vdctx->get_proc_address = hwctx->get_proc_address;



    if (level < 0)

        return AVERROR(ENOTSUP);



    status = vdctx->get_proc_address(vdctx->device,

                                     VDP_FUNC_ID_VIDEO_SURFACE_QUERY_CAPABILITIES,

                                     &func);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);

    else

        surface_query_caps = func;



    status = surface_query_caps(vdctx->device, VDP_CHROMA_TYPE_420, &supported,

                                &max_width, &max_height);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);

    if (supported != VDP_TRUE ||

        max_width < width || max_height < height)

        return AVERROR(ENOTSUP);



    status = vdctx->get_proc_address(vdctx->device,

                                     VDP_FUNC_ID_DECODER_QUERY_CAPABILITIES,

                                     &func);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);

    else

        decoder_query_caps = func;



    status = decoder_query_caps(vdctx->device, profile, &supported, &max_level,

                                &max_mb, &max_width, &max_height);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);



    if (supported != VDP_TRUE || max_level < level ||

        max_width < width || max_height < height)

        return AVERROR(ENOTSUP);



    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_CREATE,

                                     &func);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);

    else

        create = func;



    status = vdctx->get_proc_address(vdctx->device, VDP_FUNC_ID_DECODER_RENDER,

                                     &func);

    if (status != VDP_STATUS_OK)

        return vdpau_error(status);

    else

        vdctx->render = func;



    status = create(vdctx->device, profile, width, height, avctx->refs,

                    &vdctx->decoder);

    if (status == VDP_STATUS_OK) {

        vdctx->width  = avctx->coded_width;

        vdctx->height = avctx->coded_height;

    }



    return vdpau_error(status);

}
