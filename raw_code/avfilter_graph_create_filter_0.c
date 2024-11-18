int avfilter_graph_create_filter(AVFilterContext **filt_ctx, AVFilter *filt,

                                 const char *name, const char *args, void *opaque,

                                 AVFilterGraph *graph_ctx)

{

    int ret;



    *filt_ctx = avfilter_graph_alloc_filter(graph_ctx, filt, name);

    if (!*filt_ctx)

        return AVERROR(ENOMEM);



    ret = avfilter_init_filter(*filt_ctx, args, opaque);

    if (ret < 0)

        goto fail;



    return 0;



fail:

    if (*filt_ctx)

        avfilter_free(*filt_ctx);

    *filt_ctx = NULL;

    return ret;

}
