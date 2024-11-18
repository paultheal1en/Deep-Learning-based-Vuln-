static int dxva_get_decoder_guid(AVCodecContext *avctx, void *service, void *surface_format,

                                 unsigned guid_count, const GUID *guid_list, GUID *decoder_guid)

{

    FFDXVASharedContext *sctx = DXVA_SHARED_CONTEXT(avctx);

    unsigned i, j;



    *decoder_guid = ff_GUID_NULL;

    for (i = 0; dxva_modes[i].guid; i++) {

        const dxva_mode *mode = &dxva_modes[i];

        int validate;

        if (mode->codec != avctx->codec_id)

            continue;



        for (j = 0; j < guid_count; j++) {

            if (IsEqualGUID(mode->guid, &guid_list[j]))

                break;

        }

        if (j == guid_count)

            continue;



#if CONFIG_D3D11VA

        if (sctx->pix_fmt == AV_PIX_FMT_D3D11)

            validate = d3d11va_validate_output(service, *mode->guid, surface_format);

#endif

#if CONFIG_DXVA2

        if (sctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

            validate = dxva2_validate_output(service, *mode->guid, surface_format);

#endif

        if (validate) {

            *decoder_guid = *mode->guid;

            break;

        }

    }



    if (IsEqualGUID(decoder_guid, &ff_GUID_NULL)) {

        av_log(avctx, AV_LOG_VERBOSE, "No decoder device for codec found\n");

        return AVERROR(EINVAL);

    }



    if (IsEqualGUID(decoder_guid, &ff_DXVADDI_Intel_ModeH264_E))

        sctx->workaround |= FF_DXVA2_WORKAROUND_INTEL_CLEARVIDEO;



    return 0;

}
