static int configure_input_audio_filter(FilterGraph *fg, InputFilter *ifilter,

                                        AVFilterInOut *in)

{

    AVFilterContext *last_filter;

    const AVFilter *abuffer_filt = avfilter_get_by_name("abuffer");

    InputStream *ist = ifilter->ist;

    InputFile     *f = input_files[ist->file_index];

    char args[255], name[255];

    int ret, pad_idx = 0;



    snprintf(args, sizeof(args), "time_base=%d/%d:sample_rate=%d:sample_fmt=%s"

             ":channel_layout=0x%"PRIx64,

             1, ist->st->codec->sample_rate,

             ist->st->codec->sample_rate,

             av_get_sample_fmt_name(ist->st->codec->sample_fmt),

             ist->st->codec->channel_layout);

    snprintf(name, sizeof(name), "graph %d input from stream %d:%d", fg->index,

             ist->file_index, ist->st->index);



    if ((ret = avfilter_graph_create_filter(&ifilter->filter, abuffer_filt,

                                            name, args, NULL,

                                            fg->graph)) < 0)

        return ret;

    last_filter = ifilter->filter;



    if (audio_sync_method > 0) {

        AVFilterContext *async;

        int  len = 0;



        av_log(NULL, AV_LOG_WARNING, "-async has been deprecated. Used the "

               "asyncts audio filter instead.\n");



        if (audio_sync_method > 1)

            len += snprintf(args + len, sizeof(args) - len, "compensate=1:"

                            "max_comp=%d:", audio_sync_method);

        snprintf(args + len, sizeof(args) - len, "min_delta=%f",

                 audio_drift_threshold);



        snprintf(name, sizeof(name), "graph %d audio sync for input stream %d:%d",

                 fg->index, ist->file_index, ist->st->index);

        ret = avfilter_graph_create_filter(&async,

                                           avfilter_get_by_name("asyncts"),

                                           name, args, NULL, fg->graph);

        if (ret < 0)

            return ret;



        ret = avfilter_link(last_filter, 0, async, 0);

        if (ret < 0)

            return ret;



        last_filter = async;

    }

    if (audio_volume != 256) {

        AVFilterContext *volume;



        av_log(NULL, AV_LOG_WARNING, "-vol has been deprecated. Use the volume "

               "audio filter instead.\n");



        snprintf(args, sizeof(args), "volume=%f", audio_volume / 256.0);



        snprintf(name, sizeof(name), "graph %d volume for input stream %d:%d",

                 fg->index, ist->file_index, ist->st->index);

        ret = avfilter_graph_create_filter(&volume,

                                           avfilter_get_by_name("volume"),

                                           name, args, NULL, fg->graph);

        if (ret < 0)

            return ret;



        ret = avfilter_link(last_filter, 0, volume, 0);

        if (ret < 0)

            return ret;



        last_filter = volume;

    }



    snprintf(name, sizeof(name), "trim for input stream %d:%d",

             ist->file_index, ist->st->index);

    ret = insert_trim(((f->start_time == AV_NOPTS_VALUE) || !f->accurate_seek) ?

                      AV_NOPTS_VALUE : 0, INT64_MAX, &last_filter, &pad_idx, name);

    if (ret < 0)

        return ret;



    if ((ret = avfilter_link(last_filter, 0, in->filter_ctx, in->pad_idx)) < 0)

        return ret;



    return 0;

}
