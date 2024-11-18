static int start_frame_overlay(AVFilterLink *inlink, AVFilterBufferRef *inpicref)

{

    AVFilterContext *ctx = inlink->dst;

    OverlayContext *over = ctx->priv;



    inlink->cur_buf  = NULL;


    over->overpicref = inpicref;

    over->overpicref->pts = av_rescale_q(inpicref->pts, ctx->inputs[OVERLAY]->time_base,

                                         ctx->outputs[0]->time_base);

    return 0;

}