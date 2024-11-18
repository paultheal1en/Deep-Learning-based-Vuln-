static int query_formats(AVFilterGraph *graph, AVClass *log_ctx)

{

    int i, j, ret;

    int scaler_count = 0, resampler_count = 0;

    int count_queried = 0;        /* successful calls to query_formats() */

    int count_merged = 0;         /* successful merge of formats lists */

    int count_already_merged = 0; /* lists already merged */

    int count_delayed = 0;        /* lists that need to be merged later */



    for (i = 0; i < graph->nb_filters; i++) {

        AVFilterContext *f = graph->filters[i];

        if (formats_declared(f))

            continue;

        if (f->filter->query_formats)

            ret = filter_query_formats(f);

        else

            ret = ff_default_query_formats(f);

        if (ret < 0 && ret != AVERROR(EAGAIN))

            return ret;

        /* note: EAGAIN could indicate a partial success, not counted yet */

        count_queried += ret >= 0;

    }



    /* go through and merge as many format lists as possible */

    for (i = 0; i < graph->nb_filters; i++) {

        AVFilterContext *filter = graph->filters[i];



        for (j = 0; j < filter->nb_inputs; j++) {

            AVFilterLink *link = filter->inputs[j];

            int convert_needed = 0;



            if (!link)

                continue;



            if (link->in_formats != link->out_formats

                && link->in_formats && link->out_formats)

                if (!can_merge_formats(link->in_formats, link->out_formats,

                                      link->type, 0))

                    convert_needed = 1;

            if (link->type == AVMEDIA_TYPE_AUDIO) {

                if (link->in_samplerates != link->out_samplerates

                    && link->in_samplerates && link->out_samplerates)

                    if (!can_merge_formats(link->in_samplerates,

                                           link->out_samplerates,

                                           0, 1))

                        convert_needed = 1;

            }



#define MERGE_DISPATCH(field, statement)                                     \

            if (!(link->in_ ## field && link->out_ ## field)) {              \

                count_delayed++;                                             \

            } else if (link->in_ ## field == link->out_ ## field) {          \

                count_already_merged++;                                      \

            } else if (!convert_needed) {                                    \

                count_merged++;                                              \

                statement                                                    \

            }



            if (link->type == AVMEDIA_TYPE_AUDIO) {

                MERGE_DISPATCH(channel_layouts,

                    if (!ff_merge_channel_layouts(link->in_channel_layouts,

                                                  link->out_channel_layouts))

                        convert_needed = 1;

                )

                MERGE_DISPATCH(samplerates,

                    if (!ff_merge_samplerates(link->in_samplerates,

                                              link->out_samplerates))

                        convert_needed = 1;

                )

            }

            MERGE_DISPATCH(formats,

                if (!ff_merge_formats(link->in_formats, link->out_formats,

                                      link->type))

                    convert_needed = 1;

            )

#undef MERGE_DISPATCH



            if (convert_needed) {

                AVFilterContext *convert;

                AVFilter *filter;

                AVFilterLink *inlink, *outlink;

                char scale_args[256];

                char inst_name[30];



                /* couldn't merge format lists. auto-insert conversion filter */

                switch (link->type) {

                case AVMEDIA_TYPE_VIDEO:

                    if (!(filter = avfilter_get_by_name("scale"))) {

                        av_log(log_ctx, AV_LOG_ERROR, "'scale' filter "

                               "not present, cannot convert pixel formats.\n");

                        return AVERROR(EINVAL);

                    }



                    snprintf(inst_name, sizeof(inst_name), "auto-inserted scaler %d",

                             scaler_count++);



                    if ((ret = avfilter_graph_create_filter(&convert, filter,

                                                            inst_name, graph->scale_sws_opts, NULL,

                                                            graph)) < 0)

                        return ret;

                    break;

                case AVMEDIA_TYPE_AUDIO:

                    if (!(filter = avfilter_get_by_name("aresample"))) {

                        av_log(log_ctx, AV_LOG_ERROR, "'aresample' filter "

                               "not present, cannot convert audio formats.\n");

                        return AVERROR(EINVAL);

                    }



                    snprintf(inst_name, sizeof(inst_name), "auto-inserted resampler %d",

                             resampler_count++);

                    scale_args[0] = '\0';

                    if (graph->aresample_swr_opts)

                        snprintf(scale_args, sizeof(scale_args), "%s",

                                 graph->aresample_swr_opts);

                    if ((ret = avfilter_graph_create_filter(&convert, filter,

                                                            inst_name, graph->aresample_swr_opts,

                                                            NULL, graph)) < 0)

                        return ret;

                    break;

                default:

                    return AVERROR(EINVAL);

                }



                if ((ret = avfilter_insert_filter(link, convert, 0, 0)) < 0)

                    return ret;



                filter_query_formats(convert);

                inlink  = convert->inputs[0];

                outlink = convert->outputs[0];

                av_assert0( inlink-> in_formats->refcount > 0);

                av_assert0( inlink->out_formats->refcount > 0);

                av_assert0(outlink-> in_formats->refcount > 0);

                av_assert0(outlink->out_formats->refcount > 0);

                if (outlink->type == AVMEDIA_TYPE_AUDIO) {

                    av_assert0( inlink-> in_samplerates->refcount > 0);

                    av_assert0( inlink->out_samplerates->refcount > 0);

                    av_assert0(outlink-> in_samplerates->refcount > 0);

                    av_assert0(outlink->out_samplerates->refcount > 0);

                    av_assert0( inlink-> in_channel_layouts->refcount > 0);

                    av_assert0( inlink->out_channel_layouts->refcount > 0);

                    av_assert0(outlink-> in_channel_layouts->refcount > 0);

                    av_assert0(outlink->out_channel_layouts->refcount > 0);

                }

                if (!ff_merge_formats( inlink->in_formats,  inlink->out_formats,  inlink->type) ||

                    !ff_merge_formats(outlink->in_formats, outlink->out_formats, outlink->type))

                    ret = AVERROR(ENOSYS);

                if (inlink->type == AVMEDIA_TYPE_AUDIO &&

                    (!ff_merge_samplerates(inlink->in_samplerates,

                                           inlink->out_samplerates) ||

                     !ff_merge_channel_layouts(inlink->in_channel_layouts,

                                               inlink->out_channel_layouts)))

                    ret = AVERROR(ENOSYS);

                if (outlink->type == AVMEDIA_TYPE_AUDIO &&

                    (!ff_merge_samplerates(outlink->in_samplerates,

                                           outlink->out_samplerates) ||

                     !ff_merge_channel_layouts(outlink->in_channel_layouts,

                                               outlink->out_channel_layouts)))

                    ret = AVERROR(ENOSYS);



                if (ret < 0) {

                    av_log(log_ctx, AV_LOG_ERROR,

                           "Impossible to convert between the formats supported by the filter "

                           "'%s' and the filter '%s'\n", link->src->name, link->dst->name);

                    return ret;

                }

            }

        }

    }



    av_log(graph, AV_LOG_DEBUG, "query_formats: "

           "%d queried, %d merged, %d already done, %d delayed\n",

           count_queried, count_merged, count_already_merged, count_delayed);

    if (count_delayed) {

        AVBPrint bp;



        /* if count_queried > 0, one filter at least did set its formats,

           that will give additional information to its neighbour;

           if count_merged > 0, one pair of formats lists at least was merged,

           that will give additional information to all connected filters;

           in both cases, progress was made and a new round must be done */

        if (count_queried || count_merged)

            return AVERROR(EAGAIN);

        av_bprint_init(&bp, 0, AV_BPRINT_SIZE_AUTOMATIC);

        for (i = 0; i < graph->nb_filters; i++)

            if (!formats_declared(graph->filters[i]))

                av_bprintf(&bp, "%s%s", bp.len ? ", " : "",

                          graph->filters[i]->name);

        av_log(graph, AV_LOG_ERROR,

               "The following filters could not choose their formats: %s\n"

               "Consider inserting the (a)format filter near their input or "

               "output.\n", bp.str);

        return AVERROR(EIO);

    }

    return 0;

}
