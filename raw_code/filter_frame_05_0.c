static int filter_frame(AVFilterLink *inlink, AVFrame *frame)

{

    AVFilterContext *ctx = inlink->dst;

    APadContext *apad = ctx->priv;



    if (apad->whole_len)

        apad->whole_len -= frame->nb_samples;



    apad->next_pts = frame->pts + av_rescale_q(frame->nb_samples, (AVRational){1, inlink->sample_rate}, inlink->time_base);

    return ff_filter_frame(ctx->outputs[0], frame);

}
