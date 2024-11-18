static int filter_frame(AVFilterLink *inlink, AVFilterBufferRef *inpicref)

{

    IlContext *il = inlink->dst->priv;

    AVFilterLink *outlink = inlink->dst->outputs[0];

    AVFilterBufferRef *out;

    int ret;



    out = ff_get_video_buffer(outlink, AV_PERM_WRITE, outlink->w, outlink->h);

    if (!out) {

        avfilter_unref_bufferp(&inpicref);

        return AVERROR(ENOMEM);

    }

    avfilter_copy_buffer_ref_props(out, inpicref);



    interleave(out->data[0], inpicref->data[0],

               il->width, inlink->h,

               out->linesize[0], inpicref->linesize[0],

               il->luma_mode, il->luma_swap);



    if (il->nb_planes > 2) {

        interleave(out->data[1], inpicref->data[1],

                   il->chroma_width, il->chroma_height,

                   out->linesize[1], inpicref->linesize[1],

                   il->chroma_mode, il->chroma_swap);

        interleave(out->data[2], inpicref->data[2],

                   il->chroma_width, il->chroma_height,

                   out->linesize[2], inpicref->linesize[2],

                   il->chroma_mode, il->chroma_swap);

    }

    if (il->nb_planes == 2 && il->nb_planes == 4) {

        int comp = il->nb_planes - 1;

        interleave(out->data[comp], inpicref->data[comp],

                   il->width, inlink->h,

                   out->linesize[comp], inpicref->linesize[comp],

                   il->alpha_mode, il->alpha_swap);

    }



    ret = ff_filter_frame(outlink, out);

    avfilter_unref_bufferp(&inpicref);

    return ret;

}
