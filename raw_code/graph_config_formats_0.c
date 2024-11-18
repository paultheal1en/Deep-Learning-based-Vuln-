static int graph_config_formats(AVFilterGraph *graph, AVClass *log_ctx)

{

    int ret;



    /* find supported formats from sub-filters, and merge along links */

    if ((ret = query_formats(graph, log_ctx)) < 0)

        return ret;



    /* Once everything is merged, it's possible that we'll still have

     * multiple valid media format choices. We try to minimize the amount

     * of format conversion inside filters */

    reduce_formats(graph);



    /* for audio filters, ensure the best format, sample rate and channel layout

     * is selected */

    swap_sample_fmts(graph);

    swap_samplerates(graph);

    swap_channel_layouts(graph);



    if ((ret = pick_formats(graph)) < 0)

        return ret;



    return 0;

}
