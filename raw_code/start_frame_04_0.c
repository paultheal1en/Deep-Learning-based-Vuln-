static void start_frame(AVFilterLink *inlink, AVFilterBufferRef *inpicref)

{

    PadContext *pad = inlink->dst->priv;

    AVFilterBufferRef *outpicref = avfilter_ref_buffer(inpicref, ~0);

    int plane;



    inlink->dst->outputs[0]->out_buf = outpicref;



    for (plane = 0; plane < 4 && outpicref->data[plane]; plane++) {

        int hsub = (plane == 1 || plane == 2) ? pad->hsub : 0;

        int vsub = (plane == 1 || plane == 2) ? pad->vsub : 0;



        outpicref->data[plane] -= (pad->x >> hsub) * pad->line_step[plane] +

            (pad->y >> vsub) * outpicref->linesize[plane];

    }



    outpicref->video->w = pad->w;

    outpicref->video->h = pad->h;



    avfilter_start_frame(inlink->dst->outputs[0], outpicref);

}
