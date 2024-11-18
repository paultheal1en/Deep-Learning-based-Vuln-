static int filter_frame(AVFilterLink *inlink, AVFrame *ref)

{

    FrameStepContext *framestep = inlink->dst->priv;



    if (!(framestep->frame_count++ % framestep->frame_step)) {

        framestep->frame_selected = 1;

        return ff_filter_frame(inlink->dst->outputs[0], ref);

    } else {

        framestep->frame_selected = 0;

        av_frame_free(&ref);

        return 0;

    }

}
