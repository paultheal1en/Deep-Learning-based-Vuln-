static int filter_samples(AVFilterLink *inlink, AVFilterBufferRef *samplesref)

{

    AVFilterContext *ctx = inlink->dst;

    int i, ret = 0;



    for (i = 0; i < ctx->nb_outputs; i++) {

        ret = ff_filter_samples(inlink->dst->outputs[i],

                                avfilter_ref_buffer(samplesref, ~AV_PERM_WRITE));

        if (ret < 0)

            break;

    }

    avfilter_unref_buffer(samplesref);

    return ret;

}
