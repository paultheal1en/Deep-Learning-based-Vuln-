unsigned ff_dxva2_get_surface_index(const AVCodecContext *avctx,

                                    const AVDXVAContext *ctx,

                                    const AVFrame *frame)

{

    void *surface = ff_dxva2_get_surface(frame);

    unsigned i;



    for (i = 0; i < DXVA_CONTEXT_COUNT(avctx, ctx); i++)

        if (DXVA_CONTEXT_SURFACE(avctx, ctx, i) == surface)

            return i;



    assert(0);

    return 0;

}
