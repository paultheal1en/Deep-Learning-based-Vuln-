int avfilter_graph_parse(AVFilterGraph *graph, const char *filters,

                         AVFilterInOut **open_inputs, AVFilterInOut **open_outputs,

                         void *log_ctx)

{

    int index = 0, ret;

    char chr = 0;



    AVFilterInOut *curr_inputs = NULL;



    do {

        AVFilterContext *filter;

        const char *filterchain = filters;

        filters += strspn(filters, WHITESPACES);



        if ((ret = parse_inputs(&filters, &curr_inputs, open_outputs, log_ctx)) < 0)

            goto fail;



        if ((ret = parse_filter(&filter, &filters, graph, index, log_ctx)) < 0)

            goto fail;



        if (filter->input_count == 1 && !curr_inputs && !index) {

            /* First input can be omitted if it is "[in]" */

            const char *tmp = "[in]";

            if ((ret = parse_inputs(&tmp, &curr_inputs, open_outputs, log_ctx)) < 0)

                goto fail;

        }



        if ((ret = link_filter_inouts(filter, &curr_inputs, open_inputs, log_ctx)) < 0)

            goto fail;



        if ((ret = parse_outputs(&filters, &curr_inputs, open_inputs, open_outputs,

                                 log_ctx)) < 0)

            goto fail;



        filters += strspn(filters, WHITESPACES);

        chr = *filters++;



        if (chr == ';' && curr_inputs) {

            av_log(log_ctx, AV_LOG_ERROR,

                   "Invalid filterchain containing an unlabelled output pad: \"%s\"\n",

                   filterchain);

            ret = AVERROR(EINVAL);

            goto fail;

        }

        index++;

    } while (chr == ',' || chr == ';');



    if (chr) {

        av_log(log_ctx, AV_LOG_ERROR,

               "Unable to parse graph description substring: \"%s\"\n",

               filters - 1);

        ret = AVERROR(EINVAL);

        goto fail;

    }



    if (*open_inputs && !strcmp((*open_inputs)->name, "out") && curr_inputs) {

        /* Last output can be omitted if it is "[out]" */

        const char *tmp = "[out]";

        if ((ret = parse_outputs(&tmp, &curr_inputs, open_inputs, open_outputs,

                                 log_ctx)) < 0)

            goto fail;

    }



    return 0;



 fail:

    for (; graph->filter_count > 0; graph->filter_count--)

        avfilter_free(graph->filters[graph->filter_count - 1]);

    av_freep(&graph->filters);

    avfilter_inout_free(open_inputs);

    avfilter_inout_free(open_outputs);

    avfilter_inout_free(&curr_inputs);

    return ret;

}
