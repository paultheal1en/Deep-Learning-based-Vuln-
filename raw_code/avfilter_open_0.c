int avfilter_open(AVFilterContext **filter_ctx, AVFilter *filter, const char *inst_name)

{

    AVFilterContext *ret;

    *filter_ctx = NULL;



    if (!filter)

        return AVERROR(EINVAL);



    ret = av_mallocz(sizeof(AVFilterContext));



    ret->av_class = &avfilter_class;

    ret->filter   = filter;

    ret->name     = inst_name ? av_strdup(inst_name) : NULL;

    if (filter->priv_size)

        ret->priv     = av_mallocz(filter->priv_size);



    ret->input_count  = pad_count(filter->inputs);

    if (ret->input_count) {

        ret->input_pads   = av_malloc(sizeof(AVFilterPad) * ret->input_count);

        memcpy(ret->input_pads, filter->inputs, sizeof(AVFilterPad) * ret->input_count);

        ret->inputs       = av_mallocz(sizeof(AVFilterLink*) * ret->input_count);

    }



    ret->output_count = pad_count(filter->outputs);

    if (ret->output_count) {

        ret->output_pads  = av_malloc(sizeof(AVFilterPad) * ret->output_count);

        memcpy(ret->output_pads, filter->outputs, sizeof(AVFilterPad) * ret->output_count);

        ret->outputs      = av_mallocz(sizeof(AVFilterLink*) * ret->output_count);

    }



    *filter_ctx = ret;

    return 0;

}
