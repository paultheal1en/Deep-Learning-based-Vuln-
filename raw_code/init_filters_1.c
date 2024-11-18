static int init_filters(const char *filters_descr)

{

    char args[512];

    int ret;

    AVFilter *abuffersrc  = avfilter_get_by_name("abuffer");

    AVFilter *abuffersink = avfilter_get_by_name("abuffersink");

    AVFilterInOut *outputs = avfilter_inout_alloc();

    AVFilterInOut *inputs  = avfilter_inout_alloc();

    const enum AVSampleFormat sample_fmts[] = { AV_SAMPLE_FMT_S16, -1 };

    const int packing_fmts[]                = { AVFILTER_PACKED, -1 };

    const int64_t *chlayouts                = avfilter_all_channel_layouts;

    AVABufferSinkParams *abuffersink_params;

    const AVFilterLink *outlink;



    filter_graph = avfilter_graph_alloc();



    /* buffer audio source: the decoded frames from the decoder will be inserted here. */

    if (!dec_ctx->channel_layout)

        dec_ctx->channel_layout = av_get_default_channel_layout(dec_ctx->channels);

    snprintf(args, sizeof(args), "%d:%d:0x%"PRIx64":packed",

             dec_ctx->sample_rate, dec_ctx->sample_fmt, dec_ctx->channel_layout);

    ret = avfilter_graph_create_filter(&buffersrc_ctx, abuffersrc, "in",

                                       args, NULL, filter_graph);

    if (ret < 0) {

        av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer source\n");

        return ret;

    }



    /* buffer audio sink: to terminate the filter chain. */

    abuffersink_params = av_abuffersink_params_alloc();

    abuffersink_params->sample_fmts     = sample_fmts;

    abuffersink_params->channel_layouts = chlayouts;

    abuffersink_params->packing_fmts    = packing_fmts;

    ret = avfilter_graph_create_filter(&buffersink_ctx, abuffersink, "out",

                                       NULL, abuffersink_params, filter_graph);


    if (ret < 0) {

        av_log(NULL, AV_LOG_ERROR, "Cannot create audio buffer sink\n");

        return ret;

    }



    /* Endpoints for the filter graph. */

    outputs->name       = av_strdup("in");

    outputs->filter_ctx = buffersrc_ctx;

    outputs->pad_idx    = 0;

    outputs->next       = NULL;



    inputs->name       = av_strdup("out");

    inputs->filter_ctx = buffersink_ctx;

    inputs->pad_idx    = 0;

    inputs->next       = NULL;



    if ((ret = avfilter_graph_parse(filter_graph, filter_descr,

                                    &inputs, &outputs, NULL)) < 0)

        return ret;



    if ((ret = avfilter_graph_config(filter_graph, NULL)) < 0)

        return ret;



    /* Print summary of the sink buffer

     * Note: args buffer is reused to store channel layout string */

    outlink = buffersink_ctx->inputs[0];

    av_get_channel_layout_string(args, sizeof(args), -1, outlink->channel_layout);

    av_log(NULL, AV_LOG_INFO, "Output: srate:%dHz fmt:%s chlayout:%s\n",

           (int)outlink->sample_rate,

           (char *)av_x_if_null(av_get_sample_fmt_name(outlink->format), "?"),

           args);



    return 0;

}