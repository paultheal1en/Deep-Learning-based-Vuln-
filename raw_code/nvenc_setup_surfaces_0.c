static av_cold int nvenc_setup_surfaces(AVCodecContext *avctx)

{

    NvencContext *ctx = avctx->priv_data;

    int i, res;



    ctx->surfaces = av_mallocz_array(ctx->nb_surfaces, sizeof(*ctx->surfaces));

    if (!ctx->surfaces)

        return AVERROR(ENOMEM);



    ctx->timestamp_list = av_fifo_alloc(ctx->nb_surfaces * sizeof(int64_t));

    if (!ctx->timestamp_list)

        return AVERROR(ENOMEM);



    ctx->unused_surface_queue = av_fifo_alloc(ctx->nb_surfaces * sizeof(NvencSurface*));

    if (!ctx->unused_surface_queue)

        return AVERROR(ENOMEM);



    ctx->output_surface_queue = av_fifo_alloc(ctx->nb_surfaces * sizeof(NvencSurface*));

    if (!ctx->output_surface_queue)

        return AVERROR(ENOMEM);

    ctx->output_surface_ready_queue = av_fifo_alloc(ctx->nb_surfaces * sizeof(NvencSurface*));

    if (!ctx->output_surface_ready_queue)

        return AVERROR(ENOMEM);



    res = nvenc_push_context(avctx);

    if (res < 0)

        return res;



    for (i = 0; i < ctx->nb_surfaces; i++) {

        if ((res = nvenc_alloc_surface(avctx, i)) < 0)

        {

            nvenc_pop_context(avctx);

            return res;

        }

    }



    res = nvenc_pop_context(avctx);

    if (res < 0)

        return res;



    return 0;

}
