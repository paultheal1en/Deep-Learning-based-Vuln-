static int insert_trim(OutputStream *ost, AVFilterContext **last_filter, int *pad_idx)

{

    OutputFile *of = output_files[ost->file_index];

    AVFilterGraph *graph = (*last_filter)->graph;

    AVFilterContext *ctx;

    const AVFilter *trim;

    const char *name = ost->st->codec->codec_type == AVMEDIA_TYPE_VIDEO ? "trim" : "atrim";

    char filter_name[128];

    int ret = 0;



    if (of->recording_time == INT64_MAX && !of->start_time)

        return 0;



    trim = avfilter_get_by_name(name);

    if (!trim) {

        av_log(NULL, AV_LOG_ERROR, "%s filter not present, cannot limit "

               "recording time.\n", name);

        return AVERROR_FILTER_NOT_FOUND;

    }



    snprintf(filter_name, sizeof(filter_name), "%s for output stream %d:%d",

             name, ost->file_index, ost->index);

    ctx = avfilter_graph_alloc_filter(graph, trim, filter_name);

    if (!ctx)

        return AVERROR(ENOMEM);



    if (of->recording_time != INT64_MAX) {

        ret = av_opt_set_double(ctx, "duration", (double)of->recording_time / 1e6,

                                AV_OPT_SEARCH_CHILDREN);

    }

    if (ret >= 0 && of->start_time) {

        ret = av_opt_set_double(ctx, "start", (double)of->start_time / 1e6,

                                AV_OPT_SEARCH_CHILDREN);

    }

    if (ret < 0) {

        av_log(ctx, AV_LOG_ERROR, "Error configuring the %s filter", name);

        return ret;

    }



    ret = avfilter_init_str(ctx, NULL);

    if (ret < 0)

        return ret;



    ret = avfilter_link(*last_filter, *pad_idx, ctx, 0);

    if (ret < 0)

        return ret;



    *last_filter = ctx;

    *pad_idx     = 0;

    return 0;

}
