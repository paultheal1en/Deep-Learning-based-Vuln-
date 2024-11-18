static void start_frame(AVFilterLink *inlink, AVFilterBufferRef *picref)

{

    PixdescTestContext *priv = inlink->dst->priv;

    AVFilterLink *outlink    = inlink->dst->outputs[0];

    AVFilterBufferRef *outpicref;

    int i;



    outlink->out_buf = avfilter_get_video_buffer(outlink, AV_PERM_WRITE,

                                                outlink->w, outlink->h);

    outpicref = outlink->out_buf;

    avfilter_copy_buffer_ref_props(outpicref, picref);



    for (i = 0; i < 4; i++) {

        int h = outlink->h;

        h = i == 1 || i == 2 ? h>>priv->pix_desc->log2_chroma_h : h;

        if (outpicref->data[i]) {

            uint8_t *data = outpicref->data[i] +

                (outpicref->linesize[i] > 0 ? 0 : outpicref->linesize[i] * (h-1));

            memset(data, 0, FFABS(outpicref->linesize[i]) * h);

        }

    }



    /* copy palette */

    if (priv->pix_desc->flags & PIX_FMT_PAL)

        memcpy(outpicref->data[1], outpicref->data[1], 256*4);



    avfilter_start_frame(outlink, avfilter_ref_buffer(outpicref, ~0));

}
