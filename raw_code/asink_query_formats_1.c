static int asink_query_formats(AVFilterContext *ctx)

{

    BufferSinkContext *buf = ctx->priv;

    AVFilterFormats *formats = NULL;

    AVFilterChannelLayouts *layouts = NULL;

    unsigned i;

    int ret;



    CHECK_LIST_SIZE(sample_fmts)

    CHECK_LIST_SIZE(sample_rates)

    CHECK_LIST_SIZE(channel_layouts)

    CHECK_LIST_SIZE(channel_counts)



    if (buf->sample_fmts_size) {

        for (i = 0; i < NB_ITEMS(buf->sample_fmts); i++)

            if ((ret = ff_add_format(&formats, buf->sample_fmts[i])) < 0)

                return ret;

        ff_set_common_formats(ctx, formats);

    }



    if (buf->channel_layouts_size || buf->channel_counts_size ||

        buf->all_channel_counts) {

        for (i = 0; i < NB_ITEMS(buf->channel_layouts); i++)

            if ((ret = ff_add_channel_layout(&layouts, buf->channel_layouts[i])) < 0)

                return ret;

        for (i = 0; i < NB_ITEMS(buf->channel_counts); i++)

            if ((ret = ff_add_channel_layout(&layouts, FF_COUNT2LAYOUT(buf->channel_counts[i]))) < 0)

                return ret;

        if (buf->all_channel_counts) {

            if (layouts)

                av_log(ctx, AV_LOG_WARNING,

                       "Conflicting all_channel_counts and list in options\n");

            else if (!(layouts = ff_all_channel_counts()))

                return AVERROR(ENOMEM);

        }

        ff_set_common_channel_layouts(ctx, layouts);

    }



    if (buf->sample_rates_size) {

        formats = NULL;

        for (i = 0; i < NB_ITEMS(buf->sample_rates); i++)

            if ((ret = ff_add_format(&formats, buf->sample_rates[i])) < 0)

                return ret;

        ff_set_common_samplerates(ctx, formats);

    }



    return 0;

}
