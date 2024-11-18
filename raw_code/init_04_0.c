static int init(AVFilterContext *ctx, const char *args, void *opaque)

{

    GraphContext *gctx = ctx->priv;



    if(!args)

        return 0;



    if(!(gctx->link_filter = avfilter_open(&vf_graph_dummy, NULL)))

        return -1;

    if(avfilter_init_filter(gctx->link_filter, NULL, ctx))

        goto fail;



    return graph_load_chain_from_string(ctx, args, NULL, NULL);



fail:

    avfilter_destroy(gctx->link_filter);

    return -1;

}
