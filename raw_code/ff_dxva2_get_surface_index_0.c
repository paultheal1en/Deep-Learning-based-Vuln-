unsigned ff_dxva2_get_surface_index(const AVCodecContext *avctx,

                                    const AVDXVAContext *ctx,

                                    const AVFrame *frame)

{

    void *surface = get_surface(frame);

    unsigned i;



    for (i = 0; i < DXVA_CONTEXT_COUNT(avctx, ctx); i++) {

#if CONFIG_D3D11VA

        if (avctx->pix_fmt == AV_PIX_FMT_D3D11VA_VLD && ctx->d3d11va.surface[i] == surface)

        {

            D3D11_VIDEO_DECODER_OUTPUT_VIEW_DESC viewDesc;

            ID3D11VideoDecoderOutputView_GetDesc(ctx->d3d11va.surface[i], &viewDesc);

            return viewDesc.Texture2D.ArraySlice;

        }

#endif

#if CONFIG_DXVA2

        if (avctx->pix_fmt == AV_PIX_FMT_DXVA2_VLD && ctx->dxva2.surface[i] == surface)

            return i;

#endif

    }



    assert(0);

    return 0;

}
