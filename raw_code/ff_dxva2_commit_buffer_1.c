int ff_dxva2_commit_buffer(AVCodecContext *avctx,

                           AVDXVAContext *ctx,

                           DECODER_BUFFER_DESC *dsc,

                           unsigned type, const void *data, unsigned size,

                           unsigned mb_count)

{

    void     *dxva_data;

    unsigned dxva_size;

    int      result;

    HRESULT hr;



#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD)

        hr = ID3D11VideoContext_GetDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context,

                                                 D3D11VA_CONTEXT(ctx)->decoder,

                                                 type,

                                                 &dxva_size, &dxva_data);

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

        hr = IDirectXVideoDecoder_GetBuffer(DXVA2_CONTEXT(ctx)->decoder, type,

                                            &dxva_data, &dxva_size);

#endif

    if (FAILED(hr)) {

        av_log(avctx, AV_LOG_ERROR, "Failed to get a buffer for %u: 0x%lx\n",

               type, hr);

        return -1;

    }

    if (size <= dxva_size) {

        memcpy(dxva_data, data, size);



#if CONFIG_D3D11VA

        if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD) {

            D3D11_VIDEO_DECODER_BUFFER_DESC *dsc11 = dsc;

            memset(dsc11, 0, sizeof(*dsc11));

            dsc11->BufferType           = type;

            dsc11->DataSize             = size;

            dsc11->NumMBsInBuffer       = mb_count;

        }

#endif

#if CONFIG_DXVA2

        if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD) {

            DXVA2_DecodeBufferDesc *dsc2 = dsc;

            memset(dsc2, 0, sizeof(*dsc2));

            dsc2->CompressedBufferType = type;

            dsc2->DataSize             = size;

            dsc2->NumMBsInBuffer       = mb_count;

        }

#endif



        result = 0;

    } else {

        av_log(avctx, AV_LOG_ERROR, "Buffer for type %u was too small\n", type);

        result = -1;

    }



#if CONFIG_D3D11VA

    if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD)

        hr = ID3D11VideoContext_ReleaseDecoderBuffer(D3D11VA_CONTEXT(ctx)->video_context, D3D11VA_CONTEXT(ctx)->decoder, type);

#endif

#if CONFIG_DXVA2

    if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD)

        hr = IDirectXVideoDecoder_ReleaseBuffer(DXVA2_CONTEXT(ctx)->decoder, type);

#endif

    if (FAILED(hr)) {

        av_log(avctx, AV_LOG_ERROR,

               "Failed to release buffer type %u: 0x%lx\n",

               type, hr);

        result = -1;

    }

    return result;

}
