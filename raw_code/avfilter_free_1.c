void avfilter_free(AVFilterContext *filter)

{

    int i;

    AVFilterLink *link;



    if (filter->filter->uninit)

        filter->filter->uninit(filter);



    for (i = 0; i < filter->input_count; i++) {

        if ((link = filter->inputs[i])) {

            if (link->src)

                link->src->outputs[link->srcpad - link->src->output_pads] = NULL;

            avfilter_formats_unref(&link->in_formats);

            avfilter_formats_unref(&link->out_formats);

        }

        av_freep(&link);

    }

    for (i = 0; i < filter->output_count; i++) {

        if ((link = filter->outputs[i])) {

            if (link->dst)

                link->dst->inputs[link->dstpad - link->dst->input_pads] = NULL;

            avfilter_formats_unref(&link->in_formats);

            avfilter_formats_unref(&link->out_formats);

        }

        av_freep(&link);

    }



    av_freep(&filter->name);

    av_freep(&filter->input_pads);

    av_freep(&filter->output_pads);

    av_freep(&filter->inputs);

    av_freep(&filter->outputs);

    av_freep(&filter->priv);

    av_free(filter);

}
