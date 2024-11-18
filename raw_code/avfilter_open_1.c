AVFilterContext *avfilter_open(AVFilter *filter, char *inst_name)

{

    AVFilterContext *ret = av_malloc(sizeof(AVFilterContext));



    ret->av_class = av_mallocz(sizeof(AVClass));

    ret->av_class->item_name = filter_name;

    ret->filter   = filter;

    ret->name     = inst_name ? av_strdup(inst_name) : NULL;

    ret->priv     = av_mallocz(filter->priv_size);



    ret->input_count  = pad_count(filter->inputs);

    ret->input_pads   = av_malloc(sizeof(AVFilterPad) * ret->input_count);

    memcpy(ret->input_pads, filter->inputs, sizeof(AVFilterPad)*ret->input_count);

    ret->inputs       = av_mallocz(sizeof(AVFilterLink*) * ret->input_count);



    ret->output_count = pad_count(filter->outputs);

    ret->output_pads  = av_malloc(sizeof(AVFilterPad) * ret->output_count);

    memcpy(ret->output_pads, filter->outputs, sizeof(AVFilterPad)*ret->output_count);

    ret->outputs      = av_mallocz(sizeof(AVFilterLink*) * ret->output_count);



    return ret;

}
