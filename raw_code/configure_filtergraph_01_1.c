int configure_filtergraph(FilterGraph *fg)

{

    AVFilterInOut *inputs, *outputs, *cur;

    int ret, i, simple = !fg->graph_desc;

    const char *graph_desc = simple ? fg->outputs[0]->ost->avfilter :

                                      fg->graph_desc;



    avfilter_graph_free(&fg->graph);

    if (!(fg->graph = avfilter_graph_alloc()))

        return AVERROR(ENOMEM);



    if (simple) {

        OutputStream *ost = fg->outputs[0]->ost;

        char args[512];

        AVDictionaryEntry *e = NULL;



        snprintf(args, sizeof(args), "flags=0x%X", (unsigned)ost->sws_flags);

        fg->graph->scale_sws_opts = av_strdup(args);



        args[0] = '\0';

        while ((e = av_dict_get(fg->outputs[0]->ost->resample_opts, "", e,

                                AV_DICT_IGNORE_SUFFIX))) {

            av_strlcatf(args, sizeof(args), "%s=%s:", e->key, e->value);

        }

        if (strlen(args))

            args[strlen(args) - 1] = '\0';

        fg->graph->resample_lavr_opts = av_strdup(args);

    }



    if ((ret = avfilter_graph_parse2(fg->graph, graph_desc, &inputs, &outputs)) < 0)

        return ret;



    if (simple && (!inputs || inputs->next || !outputs || outputs->next)) {

        av_log(NULL, AV_LOG_ERROR, "Simple filtergraph '%s' does not have "

               "exactly one input and output.\n", graph_desc);

        return AVERROR(EINVAL);

    }



    for (cur = inputs, i = 0; cur; cur = cur->next, i++)

        if ((ret = configure_input_filter(fg, fg->inputs[i], cur)) < 0)

            return ret;

    avfilter_inout_free(&inputs);



    for (cur = outputs, i = 0; cur; cur = cur->next, i++)

        configure_output_filter(fg, fg->outputs[i], cur);

    avfilter_inout_free(&outputs);



    if ((ret = avfilter_graph_config(fg->graph, NULL)) < 0)

        return ret;



    return 0;

}
